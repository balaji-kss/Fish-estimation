import config
import torch
import os
from model import create_model
from torchvision import transforms
import cv2
import utils
import numpy as np

def load_model(epoch_num):

    model = create_model(num_classes=config.NUM_CLASSES)
    model_path = os.path.join(config.MODEL_DIR, str(epoch_num) + '.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.DEVICE)
    model.eval()

    return model

def preprocess_image(image, pad_roi_box, input_res):
    
    rsx, rsy, rex, rey = pad_roi_box
    roi_crop = image[rsy : rey, rsx : rex]

    roi_crop = utils.make_square(roi_crop)
    org_inp = roi_crop.copy()
    image = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (input_res, input_res))
    image /= 255.0
    image = transforms.ToTensor()(image)

    return image, org_inp

def predict(model, input):

    input = input.to(config.DEVICE)    
    output = model([input])[0]
    pred_boxes = output['boxes'].data.cpu().numpy().astype('int')
    pred_scores = output['scores'].data.cpu().numpy()

    return pred_boxes, pred_scores

def draw_res(org_image, pred_boxes, scores):

    image = org_image.copy()

    for score, box in zip(scores, pred_boxes):
        image = cv2.putText(image, str(round(score, 3)), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    return image

def run_video(video_path, epoch_num, input_res, roi_box, roi_pad):

    model = load_model(epoch_num)
    cap = cv2.VideoCapture(video_path)
    
    while True:

        ret, frame = cap.read()
        if ret is False: break

        pre_frame = frame.copy()
        pad_roi_box = utils.get_pad_crop(roi_box, roi_pad, pre_frame.shape)
        input, roi_crop = preprocess_image(pre_frame, pad_roi_box, input_res)
             
        pred_boxes, pred_scores = predict(model, input)
        disp_crop = draw_res(roi_crop, pred_boxes, pred_scores)
        
        cv2.imshow('disp_crop ', disp_crop)
        cv2.waitKey(-1)

if __name__ == '__main__':

    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/videos/2068116.mp4'
    epoch_num = 16
    roi_box = [600, 527, 720, 640]
    roi_pad = 2
    input_res = 512
    run_video(video_path, epoch_num, input_res, roi_box, roi_pad)