import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
import cv2
import os
from torchvision import transforms

def make_square(image):

    h, w = image.shape[:2]
    sz = max(h, w)

    pad_image = np.zeros((sz, sz, 3), dtype='uint8')
    pad_image[:h, :w] = image
    
    return pad_image

def remap_image(image):

    image *= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)

    return image

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

        self.csv_path = os.path.join(root_dir, 'det_est_anns.csv')
        self.anns = pd.read_csv(self.csv_path)
        self.root_dir = root_dir
        self.input_res = input_res

    def get_bucket_roi(self, image, bbox, pad_range=(0.5, 0.701)):
        
        pad_min, pad_max = pad_range
        pad = np.random.uniform(pad_min, pad_max)
        H, W = image.shape[:2]

        sx, sy, ex, ey = bbox
        w, h = ex - sx, ey - sy
        cx, cy = sx + 0.5 * w, sy + 0.5 * h 
        sz = 0.5 * (w + h)

        sx, sy = cx - pad * sz, cy - pad * sz
        ex, ey = cx + pad * sz, cy + pad * sz

        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(W, ex), min(H, ey)

        sx, sy = int(sx), int(sy)
        ex, ey = int(ex), int(ey)

        crop = image[sy:ey, sx:ex]

        return make_square(crop)

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

        img_path = os.path.join(self.root_dir, self.anns.iloc[idx, 0])
        image = cv2.imread(img_path)
        bbox, fill_val = self.anns.iloc[idx, 1:-1], self.anns.iloc[idx, -1]

        bckt_crop = self.get_bucket_roi(image, bbox)
        bckt_crop = self.preprocess_image(bckt_crop)
    
        bckt_crop = transforms.ToTensor()(bckt_crop)
        fill_val = torch.as_tensor([fill_val], dtype=torch.float32)

        return bckt_crop, fill_val

if __name__ == "__main__":

    root_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train_rcnn/data/'
    input_res = 512
    dataset = FillDataset(root_dir=root_dir, input_res=input_res)

    for i in range(len(dataset)):

        image, fill_val = dataset[i]
        print(i, image.shape, fill_val.shape)

        image = image.permute(1, 2, 0).numpy()
        org_image = remap_image(image)

        print('fill val: ', fill_val)
        cv2.imshow('image ', org_image)
        cv2.waitKey(-1)
