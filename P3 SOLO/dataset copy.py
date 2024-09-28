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

    def __init__(self, paths, train=True):
        # TODO: load dataset, make mask list

        img_path = paths[0]   # HDF5 file for images
        mask_path = paths[1] # HDF5 file for masks
        label_path = paths[2]  # NPY file for labels
        bbox_path = paths[3]   # NPY file for bounding boxes

        # Load images and masks from HDF5 files
        self.images = h5py.File(img_path, "r")["data"]
        self.masks = h5py.File(mask_path, "r")["data"]
        self.labels = np.load(label_path, allow_pickle=True)
        self.bboxes = np.load(bbox_path, allow_pickle=True)

        # Create a mapping between images and their corresponding masks
        self.image_to_mask = []

        # Assuming self.labels, self.bboxes, and self.masks contain information for all objects across all images
        current_index = 0
        for i in range(len(self.images)):
            num_objects_in_image = len(
                self.labels[i]
            )  # Get the number of objects in the image
            mask_indices = list(
                range(current_index, current_index + num_objects_in_image)
            )
            self.image_to_mask.append(mask_indices)
            current_index += num_objects_in_image
        # print(current_index) # should be equal to the total number of masks = 3843

        # Make 80% of the dataset for training as per instruction
        split_index = int(0.8 * len(self.images))
        if train:
            self.images = self.images[:split_index]
            self.masks = self.masks[:split_index]
            self.labels = self.labels[:split_index]
            self.bboxes = self.bboxes[:split_index]
            self.image_to_mask = self.image_to_mask[:split_index]
        else:
            self.images = self.images[split_index:]
            self.masks = self.masks[split_index:]
            self.labels = self.labels[split_index:]
            self.bboxes = self.bboxes[split_index:]
            self.image_to_mask = self.image_to_mask[split_index]

        print("Images shape: ", self.images.shape)
        print("One image shape: ", self.images[0].shape)
        print("Masks shape: ", self.masks.shape)
        print("One mask shape: ", self.masks[0].shape)
        print("Labels shape: ", self.labels.shape)
        print("One label shape: ", self.labels[0].shape)
        print("Bounding boxes shape: ", self.bboxes.shape)
        print("One bounding box shape: ", self.bboxes[0].shape)
        print("Length of image_to_mask: ", len(self.image_to_mask))

        # Define transforms for preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # output:
    # transed_img
    # label
    # transed_mask
    # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        img = self.images[index] # img shape (H, W, C)
        label = self.labels[index]
        mask = self.masks[self.image_to_mask[index]]  # Assuming shape (H, W)
        bbox = self.bboxes[index]  # Shape (n_obj, 4)

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(
            img, mask, bbox
        )

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        return len(self.images)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
    # img: 3*300*400
    # mask: 3*300*400
    # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess

        img = torch.tensor(img, dtype=torch.float32)  # Convert image to float32 tensor and permute to (C, H, W)

        # Convert mask from numpy array to a PyTorch tensor
        mask = torch.tensor(mask, dtype=torch.float32)  # Convert mask to float32 tensor

        # Resize the image and mask to (800, 1066)
        # print("Original image shape: ", img.shape, img.unsqueeze(0).shape, img.unsqueeze(0).unsqueeze(0).shape, img.unsqueeze(0).unsqueeze(0).unsqueeze(0).shape)
        # img = img.unsqueeze(0)  # Add batch and channel dimensions -> shape becomes (1, 1, C, H, W)
        # # print(img.shape)
        # print(img)
        # img = F.interpolate(img, size=(800, 1066), mode='bilinear')

        # Ensure the mask has the correct shape (1, 1, H, W) before interpolation
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions -> shape becomes (1, 1, H, W)
        mask = F.interpolate(mask, size=(800, 1066), mode='nearest').squeeze(0).squeeze(0).byte()  # Remove batch and channel dimensions

        # Normalize each channel of the image
        img = self.normalize(img)

        # Pad the image and mask to (800, 1088)
        img = F.pad(img, (0, 22))   # Right padding
        mask = F.pad(mask, (0, 22)) # Right padding for masks

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

        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []

        # Iterate through the batch
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)

        # Stack images along the first dimension
        return (
            torch.stack(transed_img_list, dim=0),
            label_list,
            transed_mask_list,
            transed_bbox_list,
        )

    def loader(self):
        # TODO: return a dataloader

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collect_fn,
        )


## Visualize debugging
if __name__ == "__main__":
    # file path and make a list
    imgs_path = "./data/hw3_mycocodata_img_comp_zlib.h5"
    masks_path = "./data/hw3_mycocodata_mask_comp_zlib.h5"
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
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = test_build_loader.loader()

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
        for i in range(batch_size):
            ## TODO: plot images with annotations
            plt.savefig("./testfig/visualtrainset" + str(iter) + ".png")
            plt.show()

        if iter == 10:
            break
