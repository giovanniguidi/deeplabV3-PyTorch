from __future__ import print_function, division
import os
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
from preprocessing import custom_transforms as tr
import random

class DeepFashionSegmentation(Dataset):
    """
    DeepFashion dataset
    """
#    NUM_CLASSES = 14

    def __init__(self,
                 config,
#                 base_dir=config['dataset']['base_path'],
                 split='train',
                 ):
        super().__init__()
        self._base_dir = config['dataset']['base_path']
        self._image_dir = os.path.join(self._base_dir, 'train', 'image')
        self._cat_dir = os.path.join(self._base_dir, 'labels')
        self.config = config
        self.split = split

        with open(os.path.join(self._base_dir, 'train_val_test.json')) as f:
            self.full_dataset = json.load(f)

        self.images = []
        self.categories = []
        self.num_classes = self.config['network']['num_classes']

        self.shuffle_dataset()

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def shuffle_dataset(self):
        #reset lists
        self.images.clear()
        self.categories.clear()

        dataset = self.full_dataset[self.split]

        if self.split == 'train' and self.config['training']['train_on_subset']['enabled']:
            fraction = self.config['training']['train_on_subset']['dataset_fraction']

            sample = int(len(dataset) * fraction)
            dataset = random.sample(dataset, sample)

        for item in dataset:
            self.images.append(os.path.join(self._image_dir, item['image']))
            self.categories.append(os.path.join(self._cat_dir, item['annotation']))

        #be sure that total dataset size is divisible by 2
        if len(self.images) % 2 != 0:
            self.images.append(os.path.join(self._image_dir, item['image']))
            self.categories.append(os.path.join(self._cat_dir, item['annotation']))

        assert (len(self.images) == len(self.categories))
#        print(self.images[0])
#        print(len(self.images))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target}

        #for split in self.split:
        if self.split == "train":
#            print('train')
            return self.transform_tr(sample)
        elif self.split == 'val':
#            print('val')
            return self.transform_val(sample)
        elif self.split == 'test':
 #           print('in return')
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.config['image']['base_size'], crop_size=self.config['image']['crop_size']),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
#            tr.FixScaleCrop(crop_size=crop_size),
            tr.FixScaleCrop(crop_size=self.config['image']['crop_size']),
#            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def preprocess(sample, crop_size=513):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'DeepFashion2(split=' + str(self.split) + ')'
