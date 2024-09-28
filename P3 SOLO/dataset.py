## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        # Link masks to images
        super().__init__()
        # [imgs_path, masks_path, labels_path, bboxes_path]
        images_file = path[0]
        masks_file = path[1]
        labels_file = path[2]
        bbx_file = path[3]        

        self.bbx = np.load(bbx_file, allow_pickle=True)
        with h5py.File(images_file, 'r') as f:
            # Access datasets f the file
            data = f['data'][:]
            self.imgs_data = torch.tensor(data)

        self.labels = np.load(labels_file, allow_pickle=True) 

        with h5py.File(masks_file, 'r') as f:
            # Access datasets f the file
            mask_data = f['data'][:]
            mask_data = torch.tensor(mask_data)
            # mask_data = data

        # according to the label length, mask can be assigned to each image
        self.masks = []
        mask_idx = 0
        for i in range(len(self.labels)):
            curr_label = self.labels[i]
            label_len = len(curr_label)
            curr_mask = mask_data[mask_idx:mask_idx+label_len]
            mask_idx += label_len
            self.masks.append(curr_mask)
        
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        # image = torch.from_numpy(self.imgs_data[index]).float()
        image = self.imgs_data[index]
        image = image / 255.0 # normalize image to [0, 1]
        image = F.interpolate(image.unsqueeze(0), size=(800, 1066), mode='bilinear').squeeze(0)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transed_img = F.pad(image, (11, 11, 0, 0))

        label = torch.from_numpy(self.labels[index])

        curr_mask = self.masks[index]
        curr_mask = curr_mask.float()
        transed_mask = F.interpolate(curr_mask.unsqueeze(0), size=(800, 1066), mode='bilinear').squeeze(0)
        transed_mask = F.pad(transed_mask, (11, 11, 0, 0))

        w_scale = 800 / 300
        h_scale = 1088 / 400
        transed_bbox = torch.from_numpy(self.bbx[index]) #bbox should also be transformed
        transed_bbox = transed_bbox * torch.tensor([w_scale, h_scale, w_scale, h_scale])

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    
    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        print("pre_process_batch")
        img = torch.from_numpy(img).float()
        img = img / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(800, 1066), mode='bilinear').squeeze(0)
        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = F.pad(img, (11, 11, 0, 0))

        mask = torch.from_numpy(mask).float()
        mask = F.interpolate(mask.unsqueeze(0), size=(800, 1066), mode='bilinear').squeeze(0)
        mask = F.pad(mask, (11, 11, 0, 0))
        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        img = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            img.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(img, dim=0), label_list, transed_mask_list, transed_bbox_list


    def loader(self):
        # TODO: return a dataloader
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collect_fn)


## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
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
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    print("loader build is sucessful")
    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        fig, ax = plt.subplots()
        for i in range(batch_size):
            ## TODO: plot images with annotations
            ax.clear()
            plt.imshow(img[i].cpu().numpy().transpose(1, 2, 0))

            for k in range(len(bbox[i])):
                x1, y1, x2, y2 = bbox[i][k]
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
            
            plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.gca().add_patch(rect).remove()
            plt.show()

            # plot the mask
            for j in range(len(mask[i])):
                plt.imshow(mask[i][j].cpu().numpy())
                plt.savefig("./testfig/visualtrainsetmask"+str(iter)+".png")
                plt.show()
            iter += 1

        if iter == 10:
            break