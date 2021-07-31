import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class DukeLiverDataset(Dataset):
    '''
    Dataset object for Duke Liver Segmentation Dataset
    '''
    def __init__(self, img_dir, annotation_file, transform = None, target_transform = None):
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
        img_path = os.path.join(self.img_dir,img_name)
        img = read_image(img_path)
        labels = []

        for obj_coordinate in annotation["coordinates"]:
            center = obj_coordinate["center"]
            polar_coordinate = obj_coordinate["polar_coordinate"]
            label = []
            label.extend(center)
            for pc in polar_coordinate:
                label.extend(pc)
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