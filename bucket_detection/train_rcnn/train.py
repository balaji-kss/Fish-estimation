# import the necessary packages
from dataset import BucketDataset
import config
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import time
import os
from model import create_model
import cv2
import utils

def data_loader(input_res):

    train_dataset = BucketDataset(root_dir=config.root_dir, input_res=input_res)

    print("[INFO] total training samples: {}...".format(len(train_dataset)), flush=True)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
	shuffle=True, num_workers=0, collate_fn=collate_fn)

    test_loader = DataLoader(train_dataset, batch_size=1,
	shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, test_loader

def train():

    model = create_model(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    train_loader, _ = data_loader(input_res)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    print("[INFO] training the network...", flush=True)

    for epoch in range(config.NUM_EPOCHS + 1):

        tloss_value, tloss_cls, tloss_box_reg, tloss_obj, tloss_rpn_box_reg = 0, 0, 0, 0, 0 
        steps = 0
        startTime = time.time()

        for images, targets in train_loader:
            
            optimizer.zero_grad()

            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_cls = loss_dict['loss_classifier']
            loss_box_reg = loss_dict['loss_box_reg']
            loss_obj = loss_dict['loss_objectness']
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
            
            tloss_value += loss_value
            tloss_cls += loss_cls
            tloss_box_reg += loss_box_reg
            tloss_obj += loss_obj
            tloss_rpn_box_reg += loss_rpn_box_reg

            losses.backward()
            optimizer.step()
            steps += 1

        endTime = time.time()	
        time_elapsed = (endTime - startTime) / 60 #mins

        avg_tloss_value = tloss_value / steps
        avg_tloss_cls = tloss_cls / steps
        avg_tloss_box_reg = tloss_box_reg / steps
        avg_tloss_obj = tloss_obj / steps
        avg_tloss_rpn_box_reg = tloss_rpn_box_reg / steps

        print("[INFO] EPOCH: {}/{}".format(epoch, config.NUM_EPOCHS), flush=True)
        print("Time: {:.3f}, Train loss: {:.6f}, Train cls loss: {:.6f}, Train box loss: {:.6f}, Train obj loss: {:.6f}, Train rpn box loss: {:.6f}".format(
		time_elapsed, avg_tloss_value, avg_tloss_cls, avg_tloss_box_reg, avg_tloss_obj, avg_tloss_rpn_box_reg), flush=True)

        if epoch % 5 == 0:
            print("[INFO] saving object detector model...", flush=True)
            model_path = os.path.join(config.MODEL_DIR, str(epoch) + '.pth') 
            torch.save(model.state_dict(), model_path)

def test():

    model = create_model(num_classes=config.NUM_CLASSES)
    model_path = os.path.join(config.MODEL_DIR, str(epoch_num) + '.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.DEVICE)
    model.eval()

    _, test_loader = data_loader(input_res)

    for images, targets in test_loader:

        images = list(image.to(config.DEVICE) for image in images)
        output = model(images)[0]

        gt_boxes = targets[0]['boxes']
        pred_boxes = output['boxes'].data.cpu().numpy().astype('int')
        scores = output['scores'].data.cpu().numpy()
        image = images[0].permute(1, 2, 0).cpu().numpy()

        image = utils.remap_image(image)
        for score, box in zip(scores, pred_boxes):
            image = cv2.putText(image, str(round(score, 3)), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

        gt_boxes = gt_boxes.cpu().numpy().astype('int')

        for gt_box in gt_boxes:
            cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 1)

        cv2.imshow('image ', image)
        cv2.waitKey(-1)

if __name__ == '__main__':

    input_res = 512
    # train()

    epoch_num = 50
    test()