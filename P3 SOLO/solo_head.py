import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
    # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
    # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
    # self.cate_out is 1 out-layer of conv2d
    # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()

        # Create 7 convolutional layers for both cate_head and ins_head
        for _ in range(self.stacked_convs):
            # Category head convolution layer
            self.cate_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.seg_feat_channels,
                        out_channels=self.seg_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                    ),
                    nn.GroupNorm(num_groups=num_groups, num_channels=self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                )
            )

            # Instance head convolution layer
            self.ins_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.seg_feat_channels,
                        out_channels=self.seg_feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                    ),
                    nn.GroupNorm(num_groups=num_groups, num_channels=self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Category head output layer
        self.cate_out = nn.Conv2d(
            in_channels=self.seg_feat_channels,
            out_channels=self.cate_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # Instance head output layer for each FPN level based on `num_grids`
        self.ins_out_list = nn.ModuleList()
        for num_grid in self.seg_num_grids:
            self.ins_out_list.append(
                nn.Conv2d(
                    in_channels=self.seg_feat_channels,
                    out_channels=num_grid ** 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                )
            )

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the

        # Initialize weights for category and instance head layers
        for layer in self.cate_head:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        for layer in self.ins_head:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        # Initialize weights for the output layers
        nn.init.xavier_uniform_(self.cate_out.weight)
        if self.cate_out.bias is not None:
            nn.init.constant_(self.cate_out.bias, 0)

        for ins_out in self.ins_out_list:
            nn.init.xavier_uniform_(ins_out.weight)
            if ins_out.bias is not None:
                nn.init.constant_(ins_out.bias, 0)

    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
    # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
    # if eval = False
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # Use MultiApply to compute the category and instance predictions in parallel for each FPN level
        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level,           # Function to forward through a single level
            new_fpn_list,                        # New FPN feature maps
            list(range(len(new_fpn_list))),      # Indexes of the FPN levels
            eval=eval,                           # Pass the eval flag
            upsample_shape=quart_shape           # Shape for upsampling in eval mode
        )

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
    # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list = []

        # Resize the first level to half its size (stride changes from 4 to 8)
        resized_feat1 = F.interpolate(fpn_feat_list[0], scale_factor=0.5, mode='bilinear', align_corners=False)
        new_fpn_list.append(resized_feat1)

        new_fpn_list.append(fpn_feat_list[1])
        new_fpn_list.append(fpn_feat_list[2])
        new_fpn_list.append(fpn_feat_list[3])

        # Resize the fifth level to double its size (stride changes from 64 to 32)
        resized_feat5 = F.interpolate(fpn_feat_list[4], scale_factor=2.0, mode='bilinear', align_corners=False)
        new_fpn_list.append(resized_feat5)

        return new_fpn_list

    # This function forward a single level of fpn_featmap through the network
    # Input:
    # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
    # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
    # if eval==False
    # cate_pred: (bz,C-1,S,S)
    # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
    # if eval==True
    # cate_pred: (bz,S,S,C-1) / after point_NMS
    # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        # Category Branch: Resize feature map using interpolate to (S, S)
        cate_feat = F.interpolate(fpn_feat, size=num_grid, mode='bilinear', align_corners=False)
        cate_pred = self.cate_head(cate_feat)
        cate_pred = self.cate_out(cate_pred)  # Output category predictions (bz, C-1, S, S)

        # Mask Branch: Concatenate coordinate information
        batch_size, _, height, width = fpn_feat.size()
        y_coords = torch.arange(height, dtype=fpn_feat.dtype, device=fpn_feat.device).view(1, 1, height, 1) / height
        x_coords = torch.arange(width, dtype=fpn_feat.dtype, device=fpn_feat.device).view(1, 1, 1, width) / width
        coord_feat = torch.cat((x_coords.repeat(batch_size, 1, 1, width), y_coords.repeat(batch_size, 1, height, 1)), dim=1)
        ins_feat = torch.cat((fpn_feat, coord_feat), dim=1)  # Concatenating with fpn_feat to make (256+2) channels
        ins_feat = self.ins_head(ins_feat)
        ins_pred = self.ins_out_list[idx](ins_feat)  # Output instance predictions (bz, S^2, 2H_feat, 2W_feat)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            # During inference, upsample the instance prediction to (Ori_H / 4, Ori_W / 4)
            ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode='bilinear', align_corners=False)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1) # from (bz,C-1,S,S) to (bz,S,S,C-1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
    # heat: (bz,C-1, S, S)
    # Output:
    # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
    # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
    # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        pass

    # This function compute the DiceLoss
    # Input:
    # mask_pred: (2H_feat, 2W_feat)
    # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
    # cate_preds: (num_entry, C-1)
    # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
    # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
    # / ins_pred_list is only used to record feature map
    # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
    # label_list: list, len(batch_size), each (n_object, )
    # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training

        # Use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list in parallel across the mini-batch
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(
            self.target_single_img,  # Function to process a single image
            ins_pred_list,           # Predictions from the mask branch
            bbox_list,               # Bounding boxes for objects in the image
            label_list,              # Labels for objects in the image
            mask_list                # Segmentation masks for objects in the image
            feature_sizes=[ins_pred.shape[-2:] for ins_pred in ins_pred_list]
        )

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
    # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
    # gt_labels_raw: n_obj,
    # gt_masks_raw: n_obj, H_ori, W_ori
    # featmap_sizes: list of shapes of featmap
    # output:
    # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
    # cate_label_list: list, len: len(FPN), (S, S)
    # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        # Iterate over the FPN levels
        for level_idx, featmap_size in enumerate(featmap_sizes):
            num_grid = self.seg_num_grids[level_idx]
            ins_label = torch.zeros(
                (num_grid**2, featmap_size[0] * 2, featmap_size[1] * 2), dtype=torch.uint8
            )
            ins_ind_label = torch.zeros((num_grid**2,), dtype=torch.uint8)
            cate_label = torch.zeros((num_grid, num_grid), dtype=torch.int64)

            # Compute the scale factor for instance size assignment
            for obj_idx in range(len(gt_labels_raw)):
                # Extract bounding box and mask
                bbox = gt_bboxes_raw[obj_idx]
                label = gt_labels_raw[obj_idx]
                mask = gt_masks_raw[obj_idx]

                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                area = w * h

                # Calculate the instance scale
                if (
                    self.scale_ranges[level_idx][0] ** 2
                    <= area
                    <= self.scale_ranges[level_idx][1] ** 2
                ):
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Find the corresponding grid cell
                    coord_x = int(center_x / featmap_size[1] * num_grid)
                    coord_y = int(center_y / featmap_size[0] * num_grid)

                    cate_label[coord_y, coord_x] = label
                    ins_ind_label[coord_y * num_grid + coord_x] = 1

                    # Resize the mask to the feature map size
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(featmap_size[0] * 2, featmap_size[1] * 2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask_resized = mask_resized.squeeze(0).squeeze(0)

                    ins_label[coord_y * num_grid + coord_x] = mask_resized.byte()

            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)
            cate_label_list.append(cate_label)

        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
    # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
    # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
    # ori_size: [ori_H, ori_W]
    # Output:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass

    # This function Postprocess on single img
    # Input:
    # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
    # cate_pred_img: (all_level_S^2, C-1)
    # Output:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
    # sorted_ins: (n_act, ori_H/4, ori_W/4)
    # sorted_scores: (n_act,)
    # Output:
    # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
    # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
    # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
    # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # color_list: list, len(C-1)
    # img: (bz,3,Ori_H, Ori_W)
    ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        
        # Iterate over each image in the batch
        for b in range(batch_size):
            fig, axs = plt.subplots(1, 5, figsize=(20, 20))

            # Recover and visualize segmentation for each of the 5 FPN levels
            for level in range(len(self.seg_num_grids)):
                ins_gt = ins_gts_list[b][level]          # (S^2, 2H_f, 2W_f)
                ins_ind_gt = ins_ind_gts_list[b][level]  # (S^2,)
                cate_gt = cate_gts_list[b][level]        # (S, S)

                num_grid = self.seg_num_grids[level]
                stride = self.strides[level]
                mask_img = np.zeros((img.shape[2], img.shape[3], 3), dtype=np.uint8)

                # Iterate over grid cells
                for i in range(num_grid):
                    for j in range(num_grid):
                        label = cate_gt[i, j]
                        if label > 0:
                            mask_index = i * num_grid + j
                            if ins_ind_gt[mask_index] > 0:
                                mask = ins_gt[mask_index].cpu().numpy()
                                mask = (mask > 0.5).astype(np.uint8)

                                x_start = j * stride
                                y_start = i * stride
                                x_end = x_start + stride
                                y_end = y_start + stride

                                mask_resized = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.uint8)
                                mask_resized[y_start:y_end, x_start:x_end] = mask

                                mask_color = plt.get_cmap(color_list[label - 1])(mask_resized)[:, :, :3]
                                mask_color = (mask_color * 255).astype(np.uint8)

                                mask_img = np.maximum(mask_img, mask_color)

                axs[level].imshow(mask_img)
                axs[level].set_title(f"FPN Level {level + 1}")
                axs[level].axis('off')

            plt.suptitle(f"Ground Truth for Image {b + 1}")
            plt.show()

    # This function plot the inference segmentation in img
    # Input:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    # color_list: ["jet", "ocean", "Spectral"]
    # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
