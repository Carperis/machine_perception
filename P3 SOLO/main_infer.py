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
        last_checkpoint = max(os.listdir(directory))
        if last_checkpoint.endswith(".pth"):

            PATH = f"{directory}/{last_checkpoint}"
            checkpoint = torch.load(PATH, weights_only=True)
            solo_head.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded model from {PATH}: epoch {checkpoint['epoch']}, loss {checkpoint['loss']}"
            )
    else:
        print("There is no valid checkpoint")
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
        # L_cate, L_mask, loss = solo_head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)

        ori_size = [img.shape[-2], img.shape[-1]]
        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list, NMS_sorted_fpn_index_list = (
            solo_head.PostProcess(ins_pred_list, cate_pred_list, ori_size)
        )

        for bz in range(batch_size):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            sorted_fpn_index = NMS_sorted_fpn_index_list[bz]
            sorted_scores = NMS_sorted_scores_list[bz]
            sorted_label = NMS_sorted_cate_label_list[bz]
            sorted_ins = NMS_sorted_ins_list[bz]
            ori_img = img[bz].permute(1, 2, 0).cpu().numpy()  # Convert image to numpy for plotting
            ori_img = (ori_img * std + mean)  # Denormalize
            ori_img = np.clip(ori_img, 0, 1)  # Ensure values are in range [0, 1]
            ori_img = (ori_img * 255).astype(np.uint8)  # Convert to 8-bit integer

            height, width = ori_img.shape[:2]
            fpn_combined_masks = {
                0: np.zeros((height, width, 3), dtype=np.uint8),
                1: np.zeros((height, width, 3), dtype=np.uint8),
                2: np.zeros((height, width, 3), dtype=np.uint8),
                3: np.zeros((height, width, 3), dtype=np.uint8),
                4: np.zeros((height, width, 3), dtype=np.uint8),
            }
            for i in range(len(sorted_fpn_index)):
                fpn_i = int(sorted_fpn_index[i])
                mask = sorted_ins[i].cpu().detach().numpy()
                label = sorted_label[i]
                score = sorted_scores[i]
                cmap = plt.colormaps.get_cmap(mask_color_list[label - 1])
                colored_mask = cmap(mask)[:, :, :3]  # Get the RGB channels
                colored_mask_processed = colored_mask[:, :, :3] * (mask > 0.5)[..., np.newaxis]
                colored_mask_resized = (colored_mask_processed * 255).astype(np.uint8)
                fpn_combined_masks[fpn_i] = np.maximum(fpn_combined_masks[fpn_i], colored_mask_resized)

            for i in range(5):
                masked_img = np.bitwise_or(ori_img, fpn_combined_masks[i])
                ax[i].imshow(masked_img)
                ax[i].set_title(f"FPN Level {i}")
                ax[i].axis("off")

        # # Visualize debugging
        # for bz in range(batch_size):
        #     for mask in NMS_sorted_ins_list[bz]:
        #         plt.figure()
        #         plt.imshow(
        #             mask.cpu().detach().numpy(),
        #             cmap="hot",
        #             interpolation="nearest",
        #         )
        #         plt.colorbar(shrink=0.5)
        #     plt.show()
        # plt.show()

        solo_head.PlotInfer(
            NMS_sorted_scores_list,
            NMS_sorted_cate_label_list,
            NMS_sorted_ins_list,
            mask_color_list,
            img
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
