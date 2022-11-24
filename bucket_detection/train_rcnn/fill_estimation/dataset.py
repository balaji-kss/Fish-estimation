import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
import cv2
import os
from torchvision import transforms
import utils
import glob
import random

class FillDataset(Dataset):
    """Fill dataset."""

    def __init__(self, root_dir, input_res = 224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_paths = glob.glob(os.path.join(root_dir, '*.csv'))
        self.root_dir = root_dir
        self.input_res = input_res
        self.populate_data()

    def populate_data(self):
        
        reform_anns = []
        for csv_path in self.csv_paths:
            print('reading ', csv_path)
            anns = pd.read_csv(csv_path)
            for i in range(len(anns)):
                imgname, ann = anns.iloc[i, 0], list(anns.iloc[i, 1:])
                reform_anns.append([imgname] + ann)

        self.anns = reform_anns

        #shuffle
        for i in range(5):
            random.shuffle(self.anns)

    def __len__(self):
        
        return len(self.anns)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_img_path, anns = self.anns[idx][0], self.anns[idx][1:] 

        img_path = os.path.join(self.root_dir, rel_img_path)
        image = cv2.imread(img_path)
        bbox, fill_val = anns[:-1], anns[-1]

        bckt_crop = utils.get_bucket_roi(image, bbox)
        bckt_crop = utils.preprocess_image(bckt_crop, self.input_res)
    
        bckt_crop = transforms.ToTensor()(bckt_crop)
        fill_val = torch.as_tensor([fill_val], dtype=torch.float32)

        return bckt_crop, fill_val

if __name__ == "__main__":

    root_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train_rcnn/data/'
    input_res = 224
    dataset = FillDataset(root_dir=root_dir, input_res=input_res)

    for i in range(len(dataset)):

        image, fill_val = dataset[i]
        print(i, image.shape, fill_val.shape)

        image = image.permute(1, 2, 0).numpy()
        org_image = utils.remap_image(image)

        print('fill val: ', fill_val)
        cv2.imshow('image ', org_image)
        cv2.waitKey(-1)
