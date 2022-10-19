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

def draw_res(org_image, pred_boxes, scores, pad_roi_box, roi_box, thresh=0.8):

    image = org_image.copy()

    cv2.rectangle(image, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 0, 0), 2)

    for score, box in zip(scores, pred_boxes):
        if score < thresh: continue
        box[0], box[2] = box[0] + pad_roi_box[0], box[2] + pad_roi_box[0]
        box[1], box[3] = box[1] + pad_roi_box[1], box[3] + pad_roi_box[1]
        image = cv2.putText(image, str(round(score, 3)), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    return image

def run_video():

    model = load_model(epoch_num)
    cap = cv2.VideoCapture(video_path)
    
    if write_video:
        out_video_path = video_path[:-4] + '_res.mp4'
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

    frameid = 0

    while True:

        ret, frame = cap.read()
        if ret is False: break

        pre_frame = frame.copy()
        pad_roi_box = utils.get_pad_crop(roi_box, roi_pad, pre_frame.shape)
        input, roi_crop = preprocess_image(pre_frame, pad_roi_box, input_res)
             
        pred_boxes, pred_scores = predict(model, input)
        disp_frame = draw_res(frame, pred_boxes, pred_scores, pad_roi_box, roi_box)
        
        # cv2.imshow('disp_frame ', disp_frame)
        # cv2.waitKey(-1)
        # print('disp_crop shape ', disp_frame.shape)

        if write_video:
            out.write(disp_frame)

        #if frameid > 1000: break

        frameid +=1

    cap.release()
    if write_video:
        out.release()

if __name__ == '__main__':

    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/videos/2068116.mp4'
    epoch_num = 46
    roi_box = [600, 530, 720, 655]
    roi_pad = 2
    input_res = 512
    write_video = 1

    run_video()