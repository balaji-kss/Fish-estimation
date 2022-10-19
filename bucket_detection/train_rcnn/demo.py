import config
import torch
import os
from model import create_model
from torchvision import transforms
import cv2
import utils
import numpy as np

def load_bckt_model(epoch_num):

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
    pred_scores = output['scores'].data.cpu().numpy().reshape((-1, 1))
    
    preds = np.concatenate((pred_scores, pred_boxes), axis=1) # score, x, y, x_, y_

    return preds.tolist()

def draw_res(org_image, roi_box, bckt_preds):

    image = org_image.copy()

    cv2.rectangle(image, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 0, 0), 2)

    for bckt_pred in bckt_preds:

        if fill_est:
            score, box, fill = bckt_pred[0], bckt_pred[1:-1], bckt_pred[-1]
            text = 'conf: ' + str(round(score, 3)) + ' fill: ' + str(fill)
        else:
            score, box, fill = bckt_pred[0], bckt_pred[1:]
            text = 'conf: ' + str(round(score, 3))
            
        image = cv2.putText(image, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
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
        score, box = bckt_box
        box[0], box[2] = box[0] + pad_roi_box[0], box[2] + pad_roi_box[0]
        box[1], box[3] = box[1] + pad_roi_box[1], box[3] + pad_roi_box[1]
        reproj_bckt_bboxes.append([score] + box)
    
    return reproj_bckt_bboxes

def detect_bucket(bckt_model, pre_frame, thresh):

    pad_roi_box = utils.get_pad_crop(roi_box, roi_pad, pre_frame.shape)
    input, roi_crop = preprocess_image(pre_frame, pad_roi_box, input_res)
    predicts = predict(bckt_model, input)
    filtered_preds = filter_bckt_preds(predicts, thresh)
    reproj_preds = reproject_bckt_bboxes(pad_roi_box, filtered_preds)

    return reproj_preds

def run_video():

    bckt_model = load_bckt_model(epoch_num)

    cap = cv2.VideoCapture(video_path)
    
    if write_video:
        out_video_path = video_path[:-4] + '_res.mp4'
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

    frameid = 0

    while True:

        ret, frame = cap.read()
        if ret is False: break

        pre_frame = frame.copy()

        bckt_dets = detect_bucket(bckt_model, pre_frame)

        # disp_frame = draw_res(frame, pred_boxes, pred_scores, pad_roi_box, roi_box)
        
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
    write_video = 0
    fill_est = 1

    run_video()