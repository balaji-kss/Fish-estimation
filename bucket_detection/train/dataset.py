import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
import cv2
import os
import config 
from torchvision import transforms
import utils

class BucketDataset(Dataset):
    """Bucket dataset."""

    def __init__(self, root_dir, input_res=224, transform=None, train=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_path = os.path.join(root_dir, 'bucket_anns.csv')
        self.anns = pd.read_csv(self.csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.input_res = input_res
        self.train = train

    def preprocess_anns(self, ann, img_shape):

        h, w = img_shape[:2]

        startX, startY, endX, endY = ann
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        return [startX, startY, endX, endY]

    def make_square(self, image):

        h, w = image.shape[:2]
        sz = max(h, w)

        pad_image = np.zeros((sz, sz, 3), dtype='uint8')
        pad_image[:h, :w] = image
        
        return pad_image

    def preprocess_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_res, self.input_res))

        return image

    def __len__(self):
        
        return len(self.anns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.anns.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = self.make_square(image)
        org_img = np.copy(image)

        bbox = self.anns.iloc[idx, 1:]

        bbox = self.preprocess_anns(bbox, image.shape)
        image = self.preprocess_image(image)
        
        bbox = np.array([bbox]).astype('float')

        if self.transform:
            image = self.transform(image)

        if self.train:
            sample = {'image': image, 'bbox': bbox}
        else:
            sample = {'image': image, 'bbox': bbox, 'original_image': org_img}

        return sample

if __name__ == "__main__":

    root_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train/data/'

    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    dataset = BucketDataset(root_dir=root_dir, transform=transforms, train=0)

    for i in range(len(dataset)):

        sample = dataset[i]
        print(i, sample['image'].shape, sample['bbox'].shape)

        image = sample['image']
        bbox = sample['bbox']
        org_image = sample['original_image']

        bbox_remap = utils.remap_bbox(bbox, org_image.shape)
        org_image = utils.draw_bbox(org_image, bbox_remap)

        cv2.imshow('image ', org_image)
        cv2.waitKey(-1)
