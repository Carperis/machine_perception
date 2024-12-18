from dataset import *
from solo_head import *


def main():
    # file path and make a list
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

    batch_size = 1
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = test_build_loader.loader()
    print("loader build is sucessful")
    mask_color_list = ["jet", "ocean", "Spectral"]

    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(
        num_classes=4
    )  ## class number is 4, because consider the background as one category.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask_list) == batch_size

        label = [label_img.to(device) for label_img in label_list]
        mask = [mask_img.to(device) for mask_img in mask_list]
        bbox = [bbox_img.to(device) for bbox_img in bbox_list]

        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())

        # plot the origin img
        fig, ax = plt.subplots()
        for i in range(batch_size):
            ## TODO: plot images with annotations
            ax.clear()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            one_img = img[i].cpu().numpy().transpose(1, 2, 0)
            img_denormalized = one_img * std + mean
            img_clipped = np.clip(
                img_denormalized, 0, 1
            )  # Ensure values are in range [0, 1]
            img_binary = (img_clipped * 255).astype(np.uint8)

            height, width = img_clipped.shape[:2]
            mask_combined = np.zeros((height, width, 3), dtype=np.uint8)

            for j in range(len(mask[i])):
                current_mask = mask[i][j].cpu().numpy().astype(int)
                cmap = plt.colormaps.get_cmap(
                    mask_color_list[label[i][j].cpu().numpy() - 1]
                )
                colored_mask = cmap(current_mask)[:, :, :3]  # Get the RGB channels
                colored_mask_processed = (
                    colored_mask[:, :, :3] * (current_mask > 0.5)[..., np.newaxis]
                )
                colored_mask_binary = (colored_mask_processed * 255).astype(np.uint8)
                mask_combined = np.maximum(mask_combined, colored_mask_binary)

            masked_img = np.bitwise_or(img_binary, mask_combined)
            plt.imshow(masked_img)

            for k in range(len(bbox[i])):
                x1, y1, x2, y2 = bbox[i][k].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=1, edgecolor="r", facecolor="none"
                )
                plt.gca().add_patch(rect)

            plt.savefig("./testfig/visualtrainset" + str(iter) + ".png")
            plt.gca().add_patch(rect).remove()
            plt.show()

            iter += 1

        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
            ins_pred_list, bbox_list, label_list, mask_list
        )
        solo_head.PlotGT(
            ins_gts_list, ins_ind_gts_list, cate_gts_list, mask_color_list, img
        )

        if iter == 10:
            break


if __name__ == "__main__":
    main()
