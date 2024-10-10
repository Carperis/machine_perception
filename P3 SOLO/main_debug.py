import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from solo_head import *
from backbone import *


def get_device():
    # Automatically select device: cuda, mps, or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# Inference script
def infer(test_dataset):
    device = get_device()
    print("Test with: ", device)

    batch_size = 4
    solo_head = SOLOHead(num_classes=4).to(device)
    solo_head.eval()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = test_build_loader.loader()

    # Load the model
    directory = "checkpoints"
    if len(os.listdir(directory)) > 0:
        last_checkpoint = os.listdir(directory)[-1]
        if last_checkpoint.endswith(".pth"):

            PATH = f"{directory}/{last_checkpoint}"
            checkpoint = torch.load(PATH, weights_only=True)
            solo_head.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {PATH}: epoch {checkpoint['epoch']}, loss {checkpoint['loss']}")
    else:
        print("There is no valid checkpoint")

    resnet50_fpn = Resnet50Backbone().to(device)

    # Inference loop
    for iter, data in enumerate(test_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)
        label_list = [label.to(device) for label in label_list]
        mask_list = [mask.to(device) for mask in mask_list]
        bbox_list = [bbox.to(device) for bbox in bbox_list]

        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
            ins_pred_list, bbox_list, label_list, mask_list
        )
        mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
        solo_head.PlotGT(ins_gts_list, ins_ind_gts_list, cate_gts_list, mask_color_list, img)

        # Plot each FPN level's grid of predicted masks and ground truth
        bz = 2  # Assuming single batch （1(12), 2(01), 3(34))
        # plt.imshow(img[bz].permute(1, 2, 0).cpu().detach().numpy())
        for fpn_idx in range(len(ins_pred_list)):  # Loop over FPN levels
            cate_pred = cate_pred_list[fpn_idx][bz]
            ins_pred = ins_pred_list[fpn_idx][bz]
            ins_gt = ins_gts_list[bz][fpn_idx]
            ins_ind_gt = ins_ind_gts_list[bz][fpn_idx]
            cate_gt = cate_gts_list[bz][fpn_idx]

            num_grid = int(np.sqrt(ins_gt.shape[0]))  # Grid size (S x S)

            fig1, axes1 = plt.subplots(num_grid, num_grid, figsize=(10, 10))
            fig1.suptitle(
                f"FPN Level {fpn_idx} - Predicted Masks", fontsize=16
            )

            # fig2, axes2 = plt.subplots(num_grid, num_grid, figsize=(10, 10))
            # fig2.suptitle(
            #     f"FPN Level {fpn_idx} - Ground Truth Masks", fontsize=16
            # )

            for grid_idx in range(num_grid**2):
                i = grid_idx // num_grid
                j = grid_idx % num_grid
                
                # if ins_ind_gt[grid_idx] == 1:
                #     pred_label = torch.sigmoid(cate_pred[i, j]).argmax().item()
                #     gt_label = cate_gt[i, j].item()
                #     print(f"Grid {grid_idx}:  GT={gt_label}, Pred={pred_label},Raw={torch.sigmoid(cate_pred[i, j])}")

                ax1 = axes1[i, j]
                pred_mask = ins_pred[grid_idx].cpu().detach().numpy()
                ax1.imshow(pred_mask, cmap="hot", interpolation="nearest")
                ax1.axis("off")

                # ax2 = axes2[i, j]
                # gt_mask = ins_gt[grid_idx].cpu().detach().numpy()
                # ax2.imshow(gt_mask, cmap="hot", interpolation="nearest")
                # ax2.axis("off")

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()


imgs_path = "./data/hw3_mycocodata_img_comp_zlib.h5"
masks_path = "./data/hw3_mycocodata_mask_comp_zlib.h5"
labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]

# Load the data into data.Dataset
dataset = BuildDataset(paths)
print("Dataset build init is successful")

# Set 20% of the dataset as the test data
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
torch.random.manual_seed(1)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

infer(test_dataset)
