import os
import torch

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


# Inference script
def infer(test_dataset):
    device = get_device()
    print("Test with: ", device)

    batch_size = 1
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
            print("This is a valid checkpoint")
            PATH = f"{directory}/{last_checkpoint}"
            checkpoint = torch.load(PATH, weights_only=True)
            solo_head.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("There is not valid checkpoint")
    resnet50_fpn = Resnet50Backbone().to(device)
    # Inferencing
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
        assert ins_pred_list[0][0].shape == ins_gts_list[0][0].shape
        # solo_head.PlotGT(ins_gts_list, ins_ind_gts_list, cate_gts_list, mask_color_list, img)

        # # Visualize debugging
        # for bz in range(batch_size):
        bz = 0
        image = img[bz].permute(1, 2, 0).cpu().detach().numpy()
        print(image.shape)
        plt.figure()
        plt.imshow(image)
        plt.title("Original Image")
        for fpn_idx in range(len(ins_pred_list)):
            # for grid_idx in range(len(ins_pred_list[fpn_idx])):
            # for grid_idx in range(len(10)):

            cate_pred = cate_pred_list[fpn_idx][bz]
            ins_pred = ins_pred_list[fpn_idx][bz]
            ins_gt = ins_gts_list[bz][fpn_idx]
            ins_ind_gt = ins_ind_gts_list[bz][fpn_idx]
            cate_gt = cate_gts_list[bz][fpn_idx]
            num_grid = int(np.sqrt(ins_gt.shape[0]))
            for grid_idx in range(num_grid**2):
                i = grid_idx // num_grid
                j = grid_idx % num_grid
                label = cate_gt[i][j]
                if ins_ind_gt[grid_idx] == 1:  # and ins_gt[grid_idx].sum() > 0:
                    gt_mask = ins_gt[grid_idx]
                    pred_mask = ins_pred[grid_idx]
                    plt.figure()
                    plt.imshow(
                        pred_mask.cpu().detach().numpy(),
                        cmap="hot",
                        interpolation="nearest",
                    )
                    plt.title(f"Mask from FPN {fpn_idx}")
                    plt.colorbar(shrink=0.5)
            # print(cate_pred.shape, ins_pred.shape)
        plt.show()

        # L_cate, L_mask, loss = solo_head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)

        ori_size = [img.shape[-2], img.shape[-1]]
        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = (
            solo_head.PostProcess(ins_pred_list, cate_pred_list, ori_size)
        )
        solo_head.PlotInfer(
            NMS_sorted_scores_list,
            NMS_sorted_cate_label_list,
            NMS_sorted_ins_list,
            mask_color_list,
            img,
            0,
        )


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

infer(test_dataset)
