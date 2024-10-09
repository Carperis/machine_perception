# Training script
import os
import torch
import matplotlib.pyplot as plt
from dataset import *
from solo_head import *
from backbone import *

def train_main(train_dataset):
    batch_size = 1
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_loader = train_build_loader.loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solo_head = SOLOHead(num_classes=4).to(device)
    num_epochs = 1
    optimizer = torch.optim.SGD(
        solo_head.parameters(),
        lr=0.01 / (16 / batch_size),
        momentum=0.9,
        weight_decay=0.0001,
    )

    resnet50_fpn = Resnet50Backbone()

    print("Training with: ", device)
    epoch = 0
    # Try loading checkpoint
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
        else:
            break

    tl = []
    fl = []
    dl = []

    for i in range(epoch, num_epochs, 1):
        print("For epoch number ", i)
        solo_head.train()
        running_loss = 0.0
        focal_loss = 0.0
        dice_loss = 0.0
        for iter, data in enumerate(train_loader, 0):
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            backout = resnet50_fpn(img)
            fpn_feat_list = list(backout.values())
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
                ins_pred_list, bbox_list, label_list, mask_list
            )
            L_cate, L_mask, loss = solo_head.loss(
                cate_pred_list,
                ins_pred_list,
                ins_gts_list,
                ins_ind_gts_list,
                cate_gts_list,
            )
            if i == 27 or i == 33:
                for group in optimizer.param_groups:
                    group["lr"] /= 10
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = +loss.item()
            dice_loss += L_mask.item()
            focal_loss += L_cate.item()
        tl.append(running_loss)
        fl.append(focal_loss)
        dl.append(dice_loss)
        # every epoch checkpoint should have a new file
        torch.save(
            {
                "epoch": i,
                "model_state_dict": solo_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
                "dice_loss": dice_loss,
                "focal_loss": focal_loss,
            },
            f"checkpoints/checkpoint_epoch_{i}.pth",
        )

        print("Total loss is ", running_loss)
        print("Dice loss is ", dice_loss)
        print("Focal loss is ", focal_loss)
    return tl, fl, dl


imgs_path = "./data/hw3_mycocodata_img_comp_zlib.h5"
masks_path = "./data/hw3_mycocodata_mask_comp_zlib.h5"
labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)
print("dataset build init is sucessful")
## Visualize debugging
# --------------------------------------------
# build the dataloader
# set 20% of the dataset as the training data
full_size = len(dataset)
print("full_size", full_size)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
# random split the dataset into training and testset
# set seed
torch.random.manual_seed(1)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
# push the randomized training data into the dataloader


total_loss, focal_loss, dice_loss = train_main(train_dataset)
plt.plot(total_loss)
plt.title("Training total loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.show()

plt.plot(focal_loss)
plt.title("Training focal loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Focal Loss")
plt.show()

plt.plot(dice_loss)
plt.title("Training dice loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.show()
