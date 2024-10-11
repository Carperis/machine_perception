import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
from datetime import datetime

from dataset import *
from solo_head import *
from backbone import *

def get_device():
    # automatically select device: cuda, mps, cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def train_main(train_dataset):
    device = get_device()
    print("Training with: ", device)

    batch_size = 1
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_loader = train_build_loader.loader()

    solo_head = SOLOHead(num_classes=4).to(device)
    num_epochs = 36
    optimizer = torch.optim.SGD(
        solo_head.parameters(),
        lr=0.01 / (16 / batch_size),
        momentum=0.9,
        weight_decay=0.0001,
    )

    resnet50_fpn = Resnet50Backbone().to(device)

    # Check if checkpoints folder exists, if not create it
    checkpoints_folder = "checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # TensorBoard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/solo_training_{current_time}")  # Specify the log directory

    epoch = 0
    tl = []
    fl = []
    dl = []
    i = 0
    while True:
        PATH = f"checkpoints/checkpoint_epoch_{i}.pth"
        i += 1
        if os.path.exists(PATH):
            checkpoint = torch.load(PATH, weights_only=True)
            solo_head.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            tl.append(loss)
            fl.append(checkpoint["focal_loss"])
            dl.append(checkpoint["dice_loss"])
            print(f"Find checkpoint from epoch {epoch}: loss={loss:.4f}, mask_loss={checkpoint['dice_loss']:.4f}, cate_loss={checkpoint['focal_loss']:.4f}")
            epoch += 1
        else:
            break

    step = 0
    accumulate_batch = 3
    loss_opt = torch.Tensor([0]).to(device)
    for i in range(epoch, num_epochs, 1):
        print("For epoch number ", i)
        solo_head.train()
        running_loss = 0.0
        focal_loss = 0.0
        dice_loss = 0.0

        # Add progress bar for training using tqdm
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader))
        counter = 0
        for iter, data in progress_bar:
            ts1 = datetime.now()
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            img = img.to(device)
            label_list = [label.to(device) for label in label_list]
            mask_list = [mask.to(device) for mask in mask_list]
            bbox_list = [bbox.to(device) for bbox in bbox_list]

            backout = resnet50_fpn(img)
            fpn_feat_list = list(backout.values())
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
            ts2 = datetime.now()
            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
                ins_pred_list, bbox_list, label_list, mask_list
            )
            ts3 = datetime.now()
            ins_gts_list = [[ins_gt.to(device) for ins_gt in ins_gt_list] for ins_gt_list in ins_gts_list]
            ins_ind_gts_list = [[ins_ind_gt.to(device) for ins_ind_gt in ins_ind_gt_list] for ins_ind_gt_list in ins_ind_gts_list]
            cate_gts_list = [[cate_gt.to(device) for cate_gt in cate_gt_list] for cate_gt_list in cate_gts_list]
            L_cate, L_mask, loss = solo_head.loss(
                cate_pred_list,
                ins_pred_list,
                ins_gts_list,
                ins_ind_gts_list,
                cate_gts_list,
            )
            ts4 = datetime.now()
            time_range_dict = {
                "forward": ts2 - ts1,
                "target": ts3 - ts2,
                "loss": ts4 - ts3,
            }
            most_time_consuming = max(time_range_dict, key=time_range_dict.get)
            # print(f"MTC: {most_time_consuming} | forward time: {ts2 - ts1}, target time: {ts3 - ts2}, loss time: {ts4 - ts3}")

            if i == 27 or i == 33:
                for group in optimizer.param_groups:
                    group["lr"] /= 10
                    # optimizer.zero_grad()
            loss_opt += loss
            # optimizer.step()

            # accumulate 4 batches and then update
            if (int(counter) + 1) % accumulate_batch == 0:
                loss_opt.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_opt = torch.Tensor([0]).to(device)

            running_loss += loss.item()
            dice_loss += L_mask.item()
            focal_loss += L_cate.item()

            normalized_running_loss = running_loss / (counter + 1)
            normalized_dice_loss = dice_loss / (counter + 1)
            normalized_focal_loss = focal_loss / (counter + 1)

            # Update progress bar with the current loss value
            progress_bar.set_description(
                f"Epoch {i}/{num_epochs} | Loss: {normalized_running_loss:.4f} | Mask Loss: {normalized_dice_loss:.4f} | Cate Loss: {normalized_focal_loss:.4f}"
            )
            # Log losses to TensorBoard
            writer.add_scalar("Loss/Total (per step)", normalized_running_loss, step)
            writer.add_scalar("Loss/Cate (per step)", normalized_focal_loss, step)
            writer.add_scalar("Loss/Mask (per step)", normalized_dice_loss, step)
            counter += 1
            step += 1

        tl.append(normalized_running_loss)
        fl.append(normalized_focal_loss)
        dl.append(normalized_dice_loss)

        writer.add_scalar("Loss/Total (per epoch)", normalized_running_loss, i)
        writer.add_scalar("Loss/Cate (per epoch)", normalized_focal_loss, i)
        writer.add_scalar("Loss/Mask (per epoch)", normalized_dice_loss, i)

        # Every epoch, save a checkpoint with a new file
        torch.save(
            {
                "epoch": i,
                "model_state_dict": solo_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": normalized_running_loss,
                "dice_loss": normalized_dice_loss,
                "focal_loss": normalized_focal_loss,
            },
            f"checkpoints/checkpoint_epoch_{i}.pth",
        )

        print("Total loss is ", normalized_running_loss)
        print("Dice loss is ", normalized_dice_loss)
        print("Focal loss is ", normalized_focal_loss)

    writer.close()  # Close the TensorBoard writer when done
    return tl, fl, dl


imgs_path = "./data/hw3_mycocodata_img_comp_zlib.h5"
masks_path = "./data/hw3_mycocodata_mask_comp_zlib.h5"
labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]

# Load the data into data.Dataset
dataset = BuildDataset(paths)
print("dataset build init is successful")

# Set 80% of the dataset as the training data
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
torch.random.manual_seed(1)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

total_loss, focal_loss, dice_loss = train_main(train_dataset)

# Plot losses using matplotlib
plt.plot(total_loss)
plt.title("Training Total Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.show()

plt.plot(focal_loss)
plt.title("Training Category Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cate Loss")
plt.show()

plt.plot(dice_loss)
plt.title("Training Mask Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Mask Loss")
plt.show()
