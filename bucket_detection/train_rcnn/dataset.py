import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
import cv2
import os
from torchvision import transforms
import utils

class BucketDataset(Dataset):
    """Bucket dataset."""

    def __init__(self, root_dir, input_res=512):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_path = os.path.join(root_dir, 'det_est_anns.csv')
        self.anns = pd.read_csv(self.csv_path)
        self.root_dir = root_dir
        self.input_res = input_res
        self.reformat_anns()

    def reformat_anns(self):

        reform_anns = {}

        for i in range(len(self.anns)):
            imgname, ann = self.anns.iloc[i, 0], self.anns.iloc[i, 1:]
            if imgname not in reform_anns:
                reform_anns[imgname] = [ann]
            else:
                reform_anns[imgname].append(ann)

        self.anns = [[key, reform_anns[key]] for key in reform_anns.keys()]

    def preprocess_boxes(self, ann, img_shape, scale=1):

        h, w = img_shape[:2]

        startX, startY, endX, endY = ann
        startX = float(startX * scale) / w
        startY = float(startY * scale) / h
        endX = float(endX * scale) / w
        endY = float(endY * scale) / h

        return [startX, startY, endX, endY]

    def preprocess_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (self.input_res, self.input_res))
        image /= 255.0
        
        return image

    def __len__(self):
        
        return len(self.anns)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_img_path, anns = self.anns[idx]

        img_path = os.path.join(self.root_dir, rel_img_path)
        image = cv2.imread(img_path)
        image = utils.make_square(image)

        bboxes = []
        for ann in anns:
            bbox = ann[:-1]
            pre_bbox = self.preprocess_boxes(bbox, image.shape, self.input_res)
            bboxes.append(pre_bbox)

        image = self.preprocess_image(image)
        
        bbox = torch.as_tensor(bboxes, dtype=torch.float32)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        
        iscrowd = torch.zeros((bbox.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor([1], dtype=torch.int64)

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        image = transforms.ToTensor()(image)

        return image, target

if __name__ == "__main__":

    root_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train_rcnn/data/'
    input_res = 512
    dataset = BucketDataset(root_dir=root_dir, input_res=input_res)

    for i in range(len(dataset)):

        image, target = dataset[i]
        print(i, image.shape, target['boxes'].shape)

        bbox = target['boxes']
        image = image.permute(1, 2, 0).numpy()
        org_image = utils.remap_image(image)
        
        bbox_remap = utils.remap_bbox(bbox, org_image.shape, scale=input_res)
        org_image = utils.draw_bbox(org_image, bbox_remap)

        cv2.imshow('image ', org_image)
        cv2.waitKey(-1)
