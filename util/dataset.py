import os
import torch
import json
import torchvision
import math
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import glob


class DukeLiverDataset(Dataset):
    '''
    Dataset object for Duke Liver Segmentation Dataset
    '''
    def __init__(self, img_dir, annotation_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.annotation_dict = self.read_json(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotation_dict["annotations"])

    def __getitem__(self, idx):
        annotations = self.annotation_dict["annotations"]
        annotation = annotations[idx]
        img_name = annotation["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path, torchvision.io.image.ImageReadMode.RGB)
        labels = []

        imgh_ori = annotation["height"]
        imgw_ori = annotation["width"]

        scalex = 224/imgw_ori
        scaley = 224/imgh_ori

        for obj_coordinate in annotation["coordinates"]:
            center = obj_coordinate["center"]
            polar_coordinate = obj_coordinate["polar_coordinate"]
            label = []
            label.extend(center)
            for pc in polar_coordinate:
                theta = pc[0]
                d = pc[1]
                dx = math.cos(theta)*d
                dy = math.sin(theta)*d
                dx_scaled = dx*scalex
                dy_scaled = dy*scaley
                d_scaled = math.sqrt(dx_scaled**2+dy_scaled**2)
                label.append(d_scaled)
            labels.append(label)
        labels = torch.tensor(labels)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            labels = self.target_transform(labels)

        return img, labels

    def read_json(self, annotation_file):
        with open(annotation_file,'r') as f:
            annotation_dict = json.load(f)
        return annotation_dict


class DukeLiverDatasetSegmentation(Dataset):
    '''
    Return the image and mask
    '''
    def __init__(self, img_dir, mask_dir, img_transform, mask_transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_file_list = []
        self.mask_file_list = []
        img_file_list = glob.glob(img_dir+"/*")
        for img_file in img_file_list:
            img_name = os.path.basename(img_file)
            mask_file = os.path.join(mask_dir, img_name)
            self.img_file_list.append(img_file)
            self.mask_file_list.append(mask_file)

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        img_file = self.img_file_list[idx]
        mask_file = self.mask_file_list[idx]

        img = torchvision.io.read_image(img_file)
        mask = torchvision.io.read_image(mask_file)

        img = img.double()
        mask = mask.double()
        img /= 255.0
        mask /= 255.0


        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask



def test_dataset():
    img_dir = "/home/data/duke_liver/dataset/fmt/imgs/ctdelay"
    annotation_file = "/home/data/duke_liver/dataset/fmt/annotation_16pts_single_obj/ctdelay_train.json"
    dataset = DukeLiverDataset(img_dir=img_dir,annotation_file=annotation_file)
    print(f"There are {len(dataset)} images in the dataset")
    datasetloader = DataLoader(dataset,batch_size=64)
    dataset_iter = iter(datasetloader)
    for img_batch,labels_batch in dataset_iter:
        print(f"image batch size: {len(img_batch)}; label batch size: {len(labels_batch)}")




if __name__ == "__main__":
    test_dataset()