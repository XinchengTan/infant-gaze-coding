import os

import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from config import *
import torch.utils.data.dataloader as dataloader


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


def get_fc_data_transforms(args, input_size, dt_key=None):

  if dt_key is not None and dt_key != 'train':
    return {dt_key: transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

  class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
      self.mean = mean
      self.std = std

    def __call__(self, tensor):
      return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
      return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

  # Apply data augmentation
  aug_list = []
  if args.cropping:
    aug_list.append(transforms.RandomResizedCrop(input_size))
  else:
    aug_list.append(transforms.Resize(input_size))
  if args.rotation:
    aug_list.append(transforms.RandomRotation(20))
  if args.color:
    aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
  if args.hor_flip:
    aug_list.append(transforms.RandomHorizontalFlip())
  if args.ver_flip:
    aug_list.append(transforms.RandomVerticalFlip())
  aug_list.append(transforms.ToTensor())
  aug_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
  if args.noise:
    aug_list.append(AddGaussianNoise(0, 0.1))
  if args.erasing:
    aug_list.append(transforms.RandomErasing())

  aug_transform = transforms.Compose(aug_list)

  # Define data transformation on train, val, test set respectively
  data_transforms = {
    'train': aug_transform,
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }
  return data_transforms


def get_dataset_dataloaders(args, input_size, batch_size, shuffle=True, num_workers=4):
  data_transforms = get_fc_data_transforms(args, input_size)

  # Create training and validation datasets
  image_datasets = {'train': datasets.ImageFolder(os.path.join(face_data_folder, 'train'), data_transforms['train']),
                    'val': datasets.ImageFolder(os.path.join(face_data_folder, 'val'), data_transforms['val']),
                    }
  # print('\n\nImageFolder class to idx: ', image_datasets['val'].class_to_idx)
  # infant - 0, target - 1
  print("# train samples:", len(image_datasets['train']))
  print("# validation samples:", len(image_datasets['val']))

  # Create training and validation dataloaders, never shuffle val and test set
  dataloaders_dict = {x: dataloader.DataLoader(image_datasets[x], batch_size=batch_size,
                                               shuffle=False if x != 'train' else shuffle,
                                               num_workers=num_workers) for x in data_transforms.keys()}
  return dataloaders_dict