import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from config import *

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class LookItDataset:
    def __init__(self, data_list, is_train):
        self.data_list = data_list
        self.is_train = is_train
        self.img_processor = data_transforms['train'] if self.is_train else data_transforms['val']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_files_seg, box_files_seg, class_seg = self.data_list[index]

        imgs = []
        for img_file in img_files_seg:
            img = Image.open(dataset_folder / img_file).convert('RGB')
            img = self.img_processor(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        boxs = []
        for box_file in box_files_seg:
            box = np.load(dataset_folder / box_file, allow_pickle=True).item()
            box = torch.tensor([box['face_size'], box['face_ver'], box['face_hor'], box['face_height'], box['face_width']])
            boxs.append(box)
        boxs = torch.stack(boxs)
        boxs = boxs.float()

        return {
            'imgs': imgs,  # n x 3 x 100 x 100
            'boxs': boxs,  # n x 5
            'label': class_seg
        }
