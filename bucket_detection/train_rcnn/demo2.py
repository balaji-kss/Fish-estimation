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

    bckt_crop = futils.get_bucket_roi(image, bckt_box, pad_range=(0.5, 0.501))

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

def draw_res(org_image, roi_box, close_idx, bckt_preds):

    image = org_image.copy()

    cv2.rectangle(image, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 0, 0), 2)

    for i in range(len(bckt_preds)):
        
        bckt_pred = bckt_preds[i]

        if close_idx is not None and i == close_idx:
            score, box, fill = bckt_pred[0], bckt_pred[1:-1], bckt_pred[-1]
            text = 'conf: ' + str(round(score, 3)) + ' fill: ' + str(round(fill, 3))
            box_color = (0, 255, 0)
            
        else:
            score, box = bckt_pred[0], bckt_pred[1:]
            text = 'conf: ' + str(round(score, 3))
            box_color = (0, 0, 255)

        box = [int(pt) for pt in box]
        image = cv2.putText(image, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1, cv2.LINE_AA)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, 2)

    return image

def filter_bckt_preds(bckt_preds, thresh):

    filter_bckt_bboxes = []
    for bckt_pred in bckt_preds:
        score = bckt_pred[0]
        if score < thresh: continue
        filter_bckt_bboxes.append(bckt_pred)
    
    return filter_bckt_bboxes

def reproject_bckt_bboxes(pad_roi_box, bckt_boxes, scale):

    reproj_bckt_bboxes = []
    for bckt_box in bckt_boxes:
        score, box = bckt_box[0], bckt_box[1:]
        box[0], box[2] = box[0] * scale + pad_roi_box[0], box[2] * scale + pad_roi_box[0]
        box[1], box[3] = box[1] * scale + pad_roi_box[1], box[3] * scale + pad_roi_box[1]
        reproj_bckt_bboxes.append([score] + box)
    
    return reproj_bckt_bboxes

def detect_bucket(bckt_model, pre_frame, thresh):

    # get pad crop
    pad_roi_box = utils.get_pad_crop(roi_box, roi_pad, pre_frame.shape)

    # preprocess input
    input, roi_crop = preprocess_bckt_image(pre_frame, pad_roi_box, input_res)
    
    # predict bucket
    predicts = predict_bckt(bckt_model, input)

    # filter based on thresh
    filtered_preds = filter_bckt_preds(predicts, thresh)

    # reproject on image
    scale = roi_crop.shape[0] / input_res
    reproj_preds = reproject_bckt_bboxes(pad_roi_box, filtered_preds, scale = scale)

    return reproj_preds, roi_crop

def is_arithmetic(nums):
    nums = sorted(nums)

    if len(nums) > 1:
        const = nums[1] - nums[0]
    else:
        return True
    for i in range(len(nums)-1):
        if nums[i+1] - nums[i] != const:
            return False
    return True

def calc_iou(box1, box2):
	
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])
	
	inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
	
	box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
	box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
	
	iou = inter_area / float(box1_area + box2_area - inter_area)

	return round(iou, 3)

def get_closest_bckt(bckt_dets, roi_box, iou_thresh):

    max_iou, cbox_idx = 0, None
    
    for i in range(len(bckt_dets)):
        bckt_box = bckt_dets[i]
        score, box = bckt_box[0], bckt_box[1:]
        iou = calc_iou(roi_box, box)
        if iou >= max_iou:
            max_iou = iou
            cbox_idx = i

    if max_iou < iou_thresh:
        cbox_idx = None

    return cbox_idx

def fillest_bound(fill_val, fill_range=(0.0, 1.0)):

    minf, maxf = fill_range[0], fill_range[1]
    fill_val = min(maxf, fill_val)
    fill_val = max(minf, fill_val)

    return fill_val

def estimate_fill(fillest_model, bckt_dets, org_frame):

    fillest_res = []

    for bckt_det in bckt_dets:
        frame = org_frame.copy()
        score, bbox = bckt_det[0], bckt_det[1:5]
        fill_input = preprocess_fillest_image(frame, bbox, fillest_input_res)
        fill_input = fill_input.to(config.DEVICE)
        predictions = fillest_model(fill_input)
        prediction = predictions[0].detach().cpu().numpy()
        fill_val = fillest_bound(float(prediction[0]))
        fillest_res.append(bckt_det + [fill_val])

    return fillest_res

def process_activity_end(no_det_frames,fills,min_detections_per_bucket,buckets,global_fills,frameid):
    
    if(len(no_det_frames)<35):
        check_frame_continuity(no_det_frames,frameid)
                
    else:
        print("Activity Finished")
        buckets,flag=record_fill(fills,min_detections_per_bucket,no_det_frames,buckets,global_fills)



def check_frame_continuity(no_det_frames,frameid):
    
    if(is_arithmetic(no_det_frames+[frameid])):
                    no_det_frames.append(frameid)

    else:
                    no_det_frames=[]

def record_fill(fills,min_detections_per_bucket,no_det_frames,buckets,global_fills):
    
    if(len(fills)>=min_detections_per_bucket):
                
                    global_fills.append(sum(fills)/len(fills))
                    buckets+=1
                    no_det_frames=[]
    flag=0
                
    fills=[]
    return buckets,flag

def run_video():

    bckt_model = load_bckt_model(bckt_epoch_num)
    fillest_model = load_fillest_model()

    cap = cv2.VideoCapture(video_path)
    
    if write_video:
        out_video_path = video_path[:-4] + '_resf.mp4'
        print('save video path: ', out_video_path)
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

    frameid = 0
    flag=0
    fills=[]
    buckets=0
    multiplier=2
    min_detections_per_bucket=6
    no_det_frames=[]
    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid +=1

        if frameid < 24000: continue
        if frameid % multiplier != 0:continue
        
        
        pre_frame = frame.copy()

        bckt_dets, roi_crop = detect_bucket(bckt_model, pre_frame, bckt_thresh)

        close_bckt_idx = get_closest_bckt(bckt_dets, roi_box, iou_thresh)
        
        #No detection for 6 seconds and we conclude that bucket has changed
        if(close_bckt_idx is None and flag==1):
            process_activity_end(no_det_frames,fills,min_detections_per_bucket,buckets,global_fills,frameid)
            
                
        
        if fill_est:
            bckt_dets = estimate_fill(fillest_model, bckt_dets, frame)
        
        
        if(close_bckt_idx is not None and flag==0):
            flag=1
            start=frameid
        
        
        if(close_bckt_idx is not None and flag==1):
            if(frameid<start+15):
                closest_bucket_fill=estimate_fill(fillest_model,[bckt_dets[close_bckt_idx]],frame)
                fills.append(closest_bucket_fill[0][-1])
                
            
            
            

            
        if display:
            disp_frame = draw_res(frame, roi_box, close_bckt_idx, bckt_dets)
            if(len(global_fills)>0):
                disp_frame = cv2.putText(disp_frame, "Total Fill: "+str(round(sum(global_fills),2)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
                disp_frame = cv2.putText(disp_frame, "Buckets: "+str(buckets), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
            else:
                disp_frame = cv2.putText(disp_frame, "Total Fill: 0", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
                disp_frame = cv2.putText(disp_frame, "Buckets: 0", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('disp_frame ', disp_frame)
            cv2.imshow('roi_crop ', roi_crop)
            cv2.waitKey(1)

        if write_video:
            out.write(disp_frame)

    cap.release()
    if write_video:
        out.release()

if __name__ == '__main__':

    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/2067840.mp4'
    bckt_epoch_num = 50
    roi_box = [600, 530, 720, 655]
    roi_pad = 2.
    input_res = 512
    fillest_input_res = 224
    write_video = 1
    fill_est = 1
    bckt_thresh = 0.2
    iou_thresh = 0.15
    display=1
    flag=0
    global_fills=[]

    run_video()
