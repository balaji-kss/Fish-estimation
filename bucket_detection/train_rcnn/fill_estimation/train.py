# import the necessary packages
from dataset import FillDataset, remap_image
import config
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import time
import os
from model import Network
import cv2
import numpy as np

def data_loader(input_res):

    train_dataset = FillDataset(root_dir=config.root_dir, input_res=input_res)

    print("[INFO] total training samples: {}...".format(len(train_dataset)), flush=True)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
	shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)

    test_loader = DataLoader(train_dataset, batch_size=1,
	shuffle=True, num_workers=1, drop_last=True)

    return train_loader, test_loader

def train():

    model = Network(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    train_loader, _ = data_loader(input_res)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.INIT_LR)
    criterion = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()

    print("[INFO] training the network...", flush=True)

    for epoch in range(config.NUM_EPOCHS + 1):

        tloss_value = 0
        steps = 0
        startTime = time.time()

        for images, gt_fill_vals in train_loader:
            
            optimizer.zero_grad()
            images = images.to(config.DEVICE)
            gt_fill_vals = gt_fill_vals.to(config.DEVICE)
            predictions = model(images)

            loss = criterion(predictions, gt_fill_vals)

            loss.backward()
            optimizer.step()
            tloss_value += loss.item()
            steps += 1

        endTime = time.time()	
        time_elapsed = (endTime - startTime) / 60 #mins
        avg_tloss_value = tloss_value / steps

        print("[INFO] EPOCH: {}/{}".format(epoch, config.NUM_EPOCHS), flush=True)
        print("Time: {:.3f} min, Train loss: {:.6f}".format(
		time_elapsed, avg_tloss_value), flush=True)

        if epoch % 5 == 0:
            print("[INFO] saving regression model...", flush=True)
            model_path = os.path.join(config.MODEL_DIR, str(epoch) + '.pth') 
            torch.save(model.state_dict(), model_path)

def test():

    model = load_fillest_model()

    _, test_loader = data_loader(input_res)

    tloss, num_imgs = 0, 0
    
    for images, gt_fill_vals in test_loader:

        images = images.to(config.DEVICE)
        predictions = model(images)

        image = images[0].permute(1, 2, 0).detach().cpu().numpy()
        prediction = predictions[0].detach().cpu().numpy()
        gt_fill_val = gt_fill_vals[0].detach().cpu().numpy()

        loss = abs(gt_fill_val - prediction)
        tloss += loss
        num_imgs += 1
        avg_loss = np.round(tloss / num_imgs, 3)

        print('gt: ', gt_fill_val, ' pred: ', prediction)
        print('loss: ', loss, ' avg loss: ', avg_loss)
        
        image = remap_image(image)
        cv2.imshow('image ', image)
        cv2.waitKey(-1)

if __name__ == '__main__':

    input_res = 224
    epoch_num = 50

    #train()
    
    test()