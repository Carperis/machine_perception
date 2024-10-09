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
        i = 2
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
                        in_channels=self.seg_feat_channels+i,
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

            i = 0

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

        # print(len(cate_pred_list), cate_pred_list[0].shape)

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
        cate_pred = F.interpolate(cate_pred, size=num_grid, mode='bilinear', align_corners=False)
        for layer in self.cate_head: cate_pred = layer(cate_pred)
        cate_pred = self.cate_out(cate_pred)  # Output category predictions (bz, C-1, S, S)

        # Mask Branch: Concatenate coordinate information
        batch_size, _, height, width = fpn_feat.size()

        # Generate coordinate information
        y_coords = torch.arange(height, dtype=fpn_feat.dtype, device=fpn_feat.device).view(1, 1, height, 1) / height  # Shape: (1, 1, height, 1)
        x_coords = torch.arange(width, dtype=fpn_feat.dtype, device=fpn_feat.device).view(1, 1, 1, width) / width  # Shape: (1, 1, height, 1)
        # Repeat x_coords and y_coords to match the dimensions of fpn_feat
        y_coords = y_coords.repeat(1, 1, 1, width)  # Shape: (1, 1, height, width)
        x_coords = x_coords.repeat(1, 1, height, 1) # Shape: (1, 1, height, width)
        # Repeat to match batch size
        y_coords = y_coords.repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, height, width)
        x_coords = x_coords.repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, 1, height, width)

        coord_feat = torch.cat((x_coords, y_coords), dim=1)
        ins_pred = torch.cat((ins_pred, coord_feat), dim=1)  # Concatenating with fpn_feat to make (256+2) channels
        for layer in self.ins_head: ins_pred = layer(ins_pred)
        ins_pred = self.ins_out_list[idx](ins_pred)  # Shape: (bz, S^2, H_feat, W_feat)

        # Output instance predictions (bz, S^2, 2H_feat, 2W_feat)
        ins_pred = F.interpolate(ins_pred, size=(height * 2, width * 2), mode='bilinear', align_corners=False)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            # During inference, upsample the instance prediction to (Ori_H / 4, Ori_W / 4)
            ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode='bilinear', align_corners=False)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1) # from (bz,C-1,S,S) to (bz,S,S,C-1)

        # check flag
        if eval == False:
            # print(cate_pred.shape,  (3, num_grid, num_grid))
            # print(ins_pred.shape, (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2))
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

        ## uniform the expression for ins_gts & ins_preds
        # ins_gts: list, len(fpn), (active_across_batch, 2H_feat, 2W_feat)
        # ins_preds: list, len(fpn), (active_across_batch, 2H_feat, 2W_feat)
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...] for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level)], 0) for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]
        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]for ins_preds_level_img, ins_ind_labels_level_img in zip(ins_preds_level, ins_ind_labels_level)], 0)for ins_preds_level, ins_ind_labels_level in zip(ins_pred_list, zip(*ins_ind_gts_list))]

        L_mask = 0
        N_pos = 0
        for ins_pred, ins_gt in zip(ins_preds, ins_gts):
            active_across_batch_size = ins_pred.size(0)
            N_pos += active_across_batch_size
            for i in range(active_across_batch_size):
                ins_pred_i = ins_pred[i]
                ins_gt_i = ins_gt[i]
                L_mask += self.DiceLoss(ins_pred_i, ins_gt_i)
        if N_pos > 0:
            L_mask = L_mask / N_pos

        ## uniform the expression for cate_gts & cate_preds
        # cate_gts: (bz*fpn*S^2,), img, fpn, grids
        # cate_preds: (bz*fpn*S^2, C-1), ([img, fpn, grids], C-1)
        cate_gts = [torch.cat([cate_gts_level_img.flatten() for cate_gts_level_img in cate_gts_level])  # cate_gts_level_img.flatten() is (S^2,)
                    for cate_gts_level in zip(*cate_gts_list)]
        cate_gts = torch.cat(cate_gts)
        cate_preds = [cate_pred_level.permute(0,2,3,1).reshape(-1, self.cate_out_channels) for cate_pred_level in cate_pred_list]
        cate_preds = torch.cat(cate_preds, 0)
        L_cate = self.FocalLoss(cate_preds, cate_gts)

        L_total = L_cate + self.mask_loss_cfg['weight'] * L_mask
        # print(L_cate, L_mask, L_total)
        return L_cate, L_mask, L_total

    # This function compute the DiceLoss
    # Input:
    # mask_pred: (2H_feat, 2W_feat)
    # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss

        # # Flatten the mask
        # mask_pred = mask_pred.contiguous().view(-1)
        # mask_gt = mask_gt.contiguous().view(-1)

        # Compute the squared terms and the Dice coefficient
        intersection = (mask_pred * mask_gt).sum()
        dice_loss = 1 - (2. * intersection) / ((mask_pred ** 2).sum() + (mask_gt ** 2).sum() + 1e-9)

        return dice_loss

    # This function compute the cate loss
    # Input:
    # cate_preds: (num_entry, C-1)
    # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        alpha = self.cate_loss_cfg['alpha']
        gamma = self.cate_loss_cfg['gamma']
        weight = self.cate_loss_cfg['weight']

        # Assuming cate_preds contains raw logits, apply sigmoid to get probabilities
        cate_preds_prob = torch.softmax(cate_preds, dim=1)  # Apply sigmoid activation

        # One-hot encode cate_gts, but ignore class 0 (background)
        cate_gts_non_zero = cate_gts - 1  # Shift class 1, 2, 3 to 0, 1, 2 (for one-hot encoding)
        cate_gts_one_hot = F.one_hot(cate_gts_non_zero.clamp(min=0), num_classes=self.cate_out_channels).float()  # Shape: (7744, 3)
        background_mask = (cate_gts == 0).unsqueeze(-1).expand_as(cate_gts_one_hot)  # Shape: (7744, 3)
        cate_gts_one_hot = cate_gts_one_hot * (~background_mask)  # Set background to [0, 0, 0]

        # Separate positive and negative cases
        pt = cate_preds_prob * cate_gts_one_hot + (1 - cate_preds_prob) * (1 - cate_gts_one_hot)  # True probability
        alpha_t = alpha * cate_gts_one_hot + (1 - alpha) * (1 - cate_gts_one_hot)  # Alpha term for balancing

        # Compute focal loss for each element
        focal_loss = -alpha_t * ((1 - pt) ** gamma) * torch.log(pt + 1e-9)  # Add small epsilon to avoid log(0)
        focal_loss = focal_loss.mean() * weight

        return focal_loss

        # cate_gts = cate_gts_one_hot.type_as(cate_preds_prob)
        # pt = (1 - cate_preds_prob) * cate_gts + cate_preds_prob * (1 - cate_gts)
        # focal_weight = (alpha * cate_gts + (1 - alpha) * (1 - cate_gts)) * pt.pow(gamma)
        # loss = F.binary_cross_entropy_with_logits(
        #     cate_preds_prob, cate_gts, reduction='none') * focal_weight
        # loss = loss.mean() * weight

        # return loss

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

        featmap_sizes = []
        for bz in range(len(bbox_list)):
            featmap_sizes.append([ins_pred.shape[-2:] for ins_pred in ins_pred_list])
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(
            self.target_single_img,  # Function to process a single image
            bbox_list,               # Bounding boxes for objects in the image
            label_list,              # Labels for objects in the image
            mask_list,                # Segmentation masks for objects in the image
            featmap_sizes
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
        for level_idx in range(len(featmap_sizes)):
            featmap_size = featmap_sizes[level_idx]
            num_grid = self.seg_num_grids[level_idx]
            ins_label = torch.zeros(
                (num_grid**2, featmap_size[0], featmap_size[1]), dtype=torch.uint8 # Shape: (S^2, 2H_feat, 2W_feat) Note: featmap_size = (2H_feat, 2W_feat)
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
                scale_check = (w * h)** (0.5)

                mask_wide = mask.shape[1]
                mask_height = mask.shape[0]

                # Calculate the instance scale
                if (
                    self.scale_ranges[level_idx][0]
                    <= scale_check
                    <= self.scale_ranges[level_idx][1]
                ):
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Determine the epsilon (scaling factor for object center region)
                    ew = w * self.epsilon
                    eh = h * self.epsilon

                    # Compute the grid coordinates
                    coord_x = int(center_x / mask_wide * num_grid)
                    coord_y = int(center_y / mask_height * num_grid)

                    # Activate the corresponding grid cell in `cate_label`
                    cate_label[coord_y, coord_x] = label
                    grid_idx = coord_y * num_grid + coord_x
                    ins_ind_label[grid_idx] = 1

                    # Resize mask to match feature map size and set the object mask
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(featmap_size[0], featmap_size[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask_resized = (mask_resized.squeeze(0).squeeze(0) > 0.5).byte()

                    # Set the mask into `ins_label` at the corresponding grid index
                    ins_label[grid_idx] = mask_resized

                    # Define the center region bounds and ensure it's within `num_grid`
                    x0, y0 = int((center_x - ew) / mask_wide * num_grid), int((center_y - eh) / mask_height * num_grid)
                    x1, y1 = int((center_x + ew) / mask_wide * num_grid), int((center_y + eh) / mask_height * num_grid)

                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    x1 = min(num_grid - 1, x1)
                    y1 = min(num_grid - 1, y1)

                    # Activate all grid cells falling inside the center region
                    for i in range(y0, y1 + 1):
                        for j in range(x0, x1 + 1):
                            if cate_label[i, j] == 0:
                                cate_label[i, j] = label
                            ins_ind_label[i * num_grid + j] = 1

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
        # Initialize the outputs
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        # Iterate over each image in the batch
        for batch_idx in range(len(cate_pred_list[0])):
            all_ins_pred = []
            all_cate_pred = []
            all_scores = []

            # Gather predictions from all levels
            for level_idx, (ins_pred_level, cate_pred_level) in enumerate(zip(ins_pred_list, cate_pred_list)):
                S = self.seg_num_grids[level_idx]
                ins_pred_img = ins_pred_level[batch_idx]  # Shape: (S^2, Ori_H/4, Ori_W/4)
                cate_pred_img = cate_pred_level[batch_idx]  # Shape: (S, S, C-1)

                # Reshape category predictions and get the scores
                cate_pred_img = cate_pred_img.permute(1, 2, 0).reshape(-1, self.cate_out_channels)  # Shape: (S^2, C-1)
                scores, cate_labels = torch.max(cate_pred_img.sigmoid(), dim=1)

                # Filter out low-confidence predictions based on category threshold
                keep = scores > self.postprocess_cfg['cate_thresh']
                scores = scores[keep]
                cate_labels = cate_labels[keep]
                ins_pred_img = ins_pred_img[keep]

                # Append predictions from this level
                all_ins_pred.append(ins_pred_img)
                all_cate_pred.append(cate_labels)
                all_scores.append(scores)

            # Concatenate predictions across all FPN levels
            all_ins_pred = torch.cat(all_ins_pred, dim=0)
            all_cate_pred = torch.cat(all_cate_pred, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            # Post-process single image predictions
            sorted_scores, sorted_cate_labels, sorted_ins = self.PostProcessImg(all_ins_pred, all_cate_pred, ori_size)

            NMS_sorted_scores_list.append(sorted_scores)
            NMS_sorted_cate_label_list.append(sorted_cate_labels)
            NMS_sorted_ins_list.append(sorted_ins)

        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list

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
        # Initialize lists to store the results
        sorted_scores = []
        sorted_cate_labels = []
        sorted_ins_masks = []

        # Sort the scores and keep the top N pre-NMS instances
        pre_NMS_num = self.postprocess_cfg['pre_NMS_num']
        sorted_indices = torch.argsort(cate_pred_img, descending=True)[:pre_NMS_num]

        # Get sorted predictions based on indices
        sorted_scores = cate_pred_img[sorted_indices]
        sorted_cate_labels = cate_pred_img[sorted_indices]
        sorted_ins_masks = ins_pred_img[sorted_indices]

        # Perform Matrix NMS to suppress overlapping masks
        decay_scores = self.MatrixNMS(sorted_ins_masks, sorted_scores)

        # Apply a threshold to keep high-confidence instances
        keep = decay_scores > self.postprocess_cfg['ins_thresh']
        sorted_scores = sorted_scores[keep]
        sorted_cate_labels = sorted_cate_labels[keep]
        sorted_ins_masks = sorted_ins_masks[keep]

        # Resize the instance masks to the original image size
        ori_H, ori_W = ori_size
        sorted_ins_masks = F.interpolate(sorted_ins_masks.unsqueeze(1), size=(ori_H, ori_W), mode='bilinear', align_corners=False).squeeze(1)

        # Keep only the top 'keep_instance' number of masks
        keep_instance = self.postprocess_cfg['keep_instance']
        if len(sorted_scores) > keep_instance:
            sorted_scores = sorted_scores[:keep_instance]
            sorted_cate_labels = sorted_cate_labels[:keep_instance]
            sorted_ins_masks = sorted_ins_masks[:keep_instance]

        return sorted_scores, sorted_cate_labels, sorted_ins_masks

    # This function perform matrix NMS
    # Input:
    # sorted_ins: (n_act, ori_H/4, ori_W/4)
    # sorted_scores: (n_act,)
    # Output:
    # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        # Flatten the masks (sorted_ins) to compute IoU
        n_act = sorted_ins.shape[0]
        sorted_ins = sorted_ins.view(n_act, -1)  # Flatten masks into (n_act, H*W)

        # Calculate intersection (element-wise multiplication)
        intersection = torch.mm(sorted_ins, sorted_ins.T)
        areas = sorted_ins.sum(dim=1)
        areas = areas.unsqueeze(1).expand_as(intersection).T
        union = areas + areas.T - intersection

        # Compute the IoU (intersection over union)
        ious = (intersection / union).triu(diagonal=1)
        ious_cmax = ious.max(dim=1)[0]
        ious_cmax = ious_cmax.expand_as(ious)

        if method == "gauss":
            decay = torch.exp(-1 * ((ious**2 - ious_cmax**2) / gauss_sigma))
        else:  # Linear decay
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        decay_scores = sorted_scores * decay

        return decay_scores

    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
    # heat: (bz,C-1, S, S)
    # Output:
    # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

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

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for bz in range(len(ins_gts_list)):
            fig, ax = plt.subplots(1, 5, figsize=(20, 4)) # 1 row, 5 columns, 20x4 inches

            # Iterate over the feature pyramid levels
            for fpn_idx in range(len(ins_gts_list[bz])):
                ins_gt = ins_gts_list[bz][fpn_idx]    # (S^2, 2H_f, 2W_f)
                ins_ind_gt = ins_ind_gts_list[bz][fpn_idx]  # (S^2,)
                cate_gt = cate_gts_list[bz][fpn_idx]  # (S, S)

                # Get the stride and the corresponding resized image
                img_resized = img[bz].permute(1, 2, 0).cpu().numpy()
                height, width = img_resized.shape[:2]

                img_resized = (img_resized * std + mean) # denormalize
                img_resized = np.clip(img_resized, 0, 1)  # Ensure values are in range [0, 1]
                img_resized_binary = (img_resized * 255).astype(np.uint8)

                # Initialize a combined mask of the same size as the original image
                mask_combined = np.zeros((height, width, 3), dtype=np.uint8)

                # Iterate over the grid cells to find active masks
                num_grid = int(np.sqrt(ins_gt.shape[0]))
                for grid_idx in range(num_grid ** 2):
                    if ins_ind_gt[grid_idx] == 1:  # If this grid cell has an object
                        mask = ins_gt[grid_idx].cpu().numpy()

                        # Calculate the grid's row and column in cate_gt
                        row = grid_idx // num_grid
                        col = grid_idx % num_grid
                        category = cate_gt[row, col].item()

                        # Use the color map corresponding to this category
                        if category > 0:
                            cmap = plt.colormaps.get_cmap(color_list[category - 1])
                            colored_mask = cmap(mask)[:, :, :3]  # Get the RGB channels
                            colored_mask_processed = colored_mask[:, :, :3] * (mask > 0.5)[..., np.newaxis]
                            # Resize the colored mask to match the original image size
                            zoom_factor_h = height / mask.shape[0]
                            zoom_factor_w = width / mask.shape[1]
                            colored_mask_resized = ndimage.zoom(colored_mask_processed, (zoom_factor_h, zoom_factor_w, 1), order=1)
                            colored_mask_resized = (colored_mask_resized * 255).astype(np.uint8)

                            mask_combined = np.maximum(mask_combined, colored_mask_resized)  # Combine masks with color

                # Display the masked image
                masked_img = np.bitwise_or(img_resized_binary, mask_combined)
                ax[fpn_idx].imshow(masked_img)
                ax[fpn_idx].set_title(f'FPN Level {fpn_idx}')
                ax[fpn_idx].axis('on')

            plt.show()

    # This function plot the inference segmentation in img
    # Input:
    # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
    # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
    # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    # color_list: ["jet", "ocean", "Spectral"]
    # img: (bz, 3, ori_H, ori_W)
    from matplotlib.colors import Normalize

    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        # Go through each image in the batch
        for bz in range(len(NMS_sorted_scores_list)):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Set up plot
            ori_img = img[bz].permute(1, 2, 0).cpu().numpy()  # Convert image to numpy for plotting
            ori_H, ori_W = ori_img.shape[:2]

            # Initialize an empty array for the combined mask (RGB)
            mask_combined = np.zeros((ori_H, ori_W, 3), dtype=np.uint8)

            # Iterate over each predicted instance
            for idx, ins_mask in enumerate(NMS_sorted_ins_list[bz]):
                score = NMS_sorted_scores_list[bz][idx]
                cate_label = NMS_sorted_cate_label_list[bz][idx]

                # Apply threshold to the mask (binary mask)
                binary_mask = (ins_mask >= 0.5).float()

                # Select a color from the color_list
                cmap = plt.get_cmap(color_list[cate_label % len(color_list)])

                # Normalize the mask and apply the colormap
                color_mask = cmap(Normalize()(binary_mask.cpu().numpy()))[:, :, :3]  # Use RGB channels only

                # Convert to the original size and overlay the mask
                color_mask = (color_mask * 255).astype(np.uint8)
                mask_combined = np.maximum(mask_combined, color_mask)

            # Overlay the mask on the original image
            masked_img = np.clip(ori_img * 255, 0, 255).astype(np.uint8)
            combined_img = 0.6 * masked_img + 0.4 * mask_combined  # Blend the image and mask

            # Plot the final result
            ax.imshow(combined_img.astype(np.uint8))
            ax.set_title(f"Inference {iter_ind}, Image {bz}")
            ax.axis("off")

            plt.show()

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
