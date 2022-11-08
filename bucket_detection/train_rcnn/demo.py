import config
import torch
import os
import sys

from model import create_model
from torchvision import transforms
import cv2
import utils
import numpy as np
from fill_estimation import utils as futils
from fill_estimation import model as fmodel

def load_bckt_model(epoch_num):

    model = create_model(num_classes=config.NUM_CLASSES)
    model_path = os.path.join(config.MODEL_DIR, str(epoch_num) + '.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.DEVICE)
    model.eval()

    return model

def load_fillest_model():
    
    model = fmodel.Network(num_classes=config.FILL_NUM_CLASSES)
    model_path = config.FILL_MODEL_PATH
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.DEVICE)
    model.eval()

    return model

def preprocess_bckt_image(image, pad_roi_box, input_res):
    
    rsx, rsy, rex, rey = pad_roi_box
    roi_crop = image[rsy : rey, rsx : rex]

    roi_crop = utils.make_square(roi_crop)
    org_inp = roi_crop.copy()
    image = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (input_res, input_res))
    image /= 255.0
    image = transforms.ToTensor()(image)

    return image, org_inp

def preprocess_fillest_image(image, bckt_box, input_res, debug = 0):

    bckt_crop = futils.get_bucket_roi(image, bckt_box)

    if debug:
        cv2.imshow('bckt_crop ', bckt_crop)
        cv2.waitKey(-1)

    bckt_crop = futils.preprocess_image(bckt_crop, input_res)
    bckt_crop = transforms.ToTensor()(bckt_crop)
    bckt_crop = torch.unsqueeze(bckt_crop, 0)

    return bckt_crop  

def predict_bckt(model, input):

    input = input.to(config.DEVICE)    
    output = model([input])[0]
    pred_boxes = output['boxes'].data.cpu().numpy().astype('int')
    pred_scores = output['scores'].data.cpu().numpy().reshape((-1, 1))
    
    preds = np.concatenate((pred_scores, pred_boxes), axis=1) # score, x, y, x_, y_

    return preds.tolist()

def draw_res(org_image, roi_box, bckt_preds):

    image = org_image.copy()

    cv2.rectangle(image, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 0, 0), 2)

    for bckt_pred in bckt_preds:

        if fill_est:
            score, box, fill = bckt_pred[0], bckt_pred[1:-1], bckt_pred[-1]
            text = 'conf: ' + str(round(score, 3)) + ' fill: ' + str(round(fill, 3))
        else:
            score, box = bckt_pred[0], bckt_pred[1:]
            text = 'conf: ' + str(round(score, 3))

        box = [int(pt) for pt in box]
        image = cv2.putText(image, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    return image

def filter_bckt_preds(bckt_preds, thresh):

    filter_bckt_bboxes = []
    for bckt_pred in bckt_preds:
        score = bckt_pred[0]
        if score < thresh: continue
        filter_bckt_bboxes.append(bckt_pred)
    
    return filter_bckt_bboxes

def reproject_bckt_bboxes(pad_roi_box, bckt_boxes):

    reproj_bckt_bboxes = []
    for bckt_box in bckt_boxes:
        score, box = bckt_box[0], bckt_box[1:]
        box[0], box[2] = box[0] + pad_roi_box[0], box[2] + pad_roi_box[0]
        box[1], box[3] = box[1] + pad_roi_box[1], box[3] + pad_roi_box[1]
        reproj_bckt_bboxes.append([score] + box)
    
    return reproj_bckt_bboxes

def detect_bucket(bckt_model, pre_frame, thresh):

    pad_roi_box = utils.get_pad_crop(roi_box, roi_pad, pre_frame.shape)
    input, roi_crop = preprocess_bckt_image(pre_frame, pad_roi_box, input_res)
    predicts = predict_bckt(bckt_model, input)
    filtered_preds = filter_bckt_preds(predicts, thresh)
    reproj_preds = reproject_bckt_bboxes(pad_roi_box, filtered_preds)

    return reproj_preds

def estimate_fill(fillest_model, bckt_dets, org_frame):

    fillest_res = []

    for bckt_det in bckt_dets:
        frame = org_frame.copy()
        score, bbox = bckt_det[0], bckt_det[1:5]
        fill_input = preprocess_fillest_image(frame, bbox, fillest_input_res)
        fill_input = fill_input.to(config.DEVICE)
        predictions = fillest_model(fill_input)
        prediction = predictions[0].detach().cpu().numpy()
        fillest_res.append(bckt_det + [float(prediction[0])])

    return fillest_res

def run_video():

    bckt_model = load_bckt_model(bckt_epoch_num)
    fillest_model = load_fillest_model()

    cap = cv2.VideoCapture(video_path)
    
    if write_video:
        out_video_path = video_path[:-4] + '_resf.mp4'
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

    frameid = 0

    while True:

        ret, frame = cap.read()
        if ret is False: break

        pre_frame = frame.copy()

        bckt_dets = detect_bucket(bckt_model, pre_frame, bckt_thresh)

        if fill_est:
            bckt_dets = estimate_fill(fillest_model, bckt_dets, frame)

        disp_frame = draw_res(frame, roi_box, bckt_dets)
        
        # cv2.imshow('disp_frame ', disp_frame)
        # cv2.waitKey(-1)

        if write_video:
            out.write(disp_frame)

        frameid +=1

    cap.release()
    if write_video:
        out.release()

if __name__ == '__main__':

    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/videos/2068016.mp4'
    bckt_epoch_num = 46
    roi_box = [600, 530, 720, 655]
    roi_pad = 2
    input_res = 512
    fillest_input_res = 224
    write_video = 1
    fill_est = 1
    bckt_thresh = 0.8

    run_video()