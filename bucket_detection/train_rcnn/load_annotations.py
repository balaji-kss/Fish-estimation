from sqlite3 import Row
import numpy as np
from imutils import paths
import config
import cv2
import os 
import csv
import utils

def populate_frames(video_path, skip_frames, start_frame, max_num_frame=1000000):

    cap = cv2.VideoCapture(video_path)
    count = 0
    frames = []

    while True:

        ret, frame = cap.read()
        if ret is False: break
        count += 1

        if count<start_frame:continue

        if count%skip_frames!=0:continue

        frames.append(frame)

        if count>max_num_frame:
            break
    
    return frames

def populate_anns(ann_dir, skip_frames):

    frame_dict = {}

    for csvPath in paths.list_files(ann_dir, validExts=(".csv")):
        # load the contents of the current CSV annotations file
        rows = open(csvPath).read().strip().split("\n")
    
        # loop over the rows
        for row in rows[1:]:
            # break the row into the filename, bounding box coordinates,
            # and class label
            row = row.split(",")
            
            row = [float(ele) for ele in row]
            
            (idx, frame_id, startX, startY, w, h, fill, video_id) = row
            idx, frame_id, startX, startY, w, h = int(idx), int(frame_id), int(startX), int(startY), int(w), int(h)
            video_id = int(video_id)
            endX, endY = startX + w, startY + h

            if frame_id%skip_frames!=0:continue
            
            if frame_id in frame_dict:
                frame_dict[frame_id].append([startX, startY, endX, endY, fill])
            else:
                frame_dict[frame_id] = [[startX, startY, endX, endY, fill]]
    
    return frame_dict

def filter_anns(anns, area_max = 1100 * 1100):

    filtered_anns = []

    for ann in anns:
        startX, startY, endX, endY, fill = ann
        w, h = endX - startX, endY - startY

        if w <= 0 or h <= 0 or w * h >= area_max:continue

        filtered_anns.append([startX, startY, endX, endY, fill])

    return filtered_anns

def pad_image(image, anns, pad_rg=(1.5, 3), debug=1):

    pad_min, pad_max = pad_rg
    pad = np.random.uniform(pad_min, pad_max)
    H, W = image.shape[:2]

    crops, crop_anns, labels = [], [], []

    for ann in anns:
        startX, startY, endX, endY = ann
        w, h = endX - startX, endY - startY
        cx, cy =  startX + 0.5 * w, startY + 0.5 * h 
        sz = 0.5 * (w + h)

        sx, sy = cx - pad * sz, cy - pad * sz
        ex, ey = cx + pad * sz, cy + pad * sz

        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(W, ex), min(H, ey)

        sx, sy = int(sx), int(sy)
        ex, ey = int(ex), int(ey)

        crop = image[sy:ey, sx:ex]
        crop_ann = [startX - sx, startY - sy, endX  - sx, endY  - sy]
        
        if debug:
            crop = utils.draw_bbox(crop, [crop_ann])        
            cv2.imshow('img ', crop)
            cv2.waitKey(-1)

        # pp_crop = preprocess_image(crop)
        # pp_crop_anns = preprocess_anns(ann, pp_crop.shape)
        crops.append(crop)
        crop_anns.append(crop_ann)
        labels.append('bucket')

    return crops, crop_anns, labels

def checkin(bbox, roi):

    bsx, bsy, bex, bey = bbox[:4]
    rsx, rsy, rex, rey = roi

    sx = 0
    if bsx >= rsx and bsx <= rex:
        sx = 1

    sy = 0
    if bsy >= rsy and bsy <= rey:
        sy = 1
    
    ex = 0
    if bex >= rsx and bex <= rex:
        ex = 1
    
    ey = 0
    if bey >= rsy and bey <= rey:
        ey = 1

    return sx and sy and ex and ey

def pad_roi(image, anns, roi, pad_rg=(2.0, 3.0), debug=1):

    pad_min, pad_max = pad_rg
    pad = np.random.uniform(pad_min, pad_max)
    H, W = image.shape[:2]

    pad_roi = utils.get_pad_crop(roi, pad, image.shape)
    rsx, rsy, rex, rey = pad_roi
    crop = image[rsy : rey, rsx : rex]
    crop_anns, labels = [], []

    for ann in anns:

        if not checkin(ann, pad_roi):continue
        sx, sy, ex, ey, fill = ann
        crop_ann = [sx - rsx, sy - rsy, ex - rsx, ey - rsy]
        crop_anns.append(crop_ann + [fill])
        labels.append('bucket')

    if debug:
        crop = utils.draw_bbox(crop, crop_anns)        
        cv2.imshow('img ', crop)
        cv2.waitKey(-1)

    print('len anns: ', len(crop_anns))
    if len(crop_anns) == 0:
        print('*****no annotation*****')
        return None, None, None

    return [crop], crop_anns, labels
        
def load_data(ann_dir, video_path, debug):

    print("[INFO] loading dataset...")
    data = []
    bboxes = []
    labels = []

    skip_frames = 1
    frame_ann_dict = populate_anns(ann_dir, skip_frames)
    start_frame = min([key for key in frame_ann_dict.keys()])
    print(start_frame)
    frames = populate_frames(video_path, skip_frames, start_frame)

    print('total frames: ', len(frames))
    print('total anns: ', len(frame_ann_dict))

    image_id = 0
    roi = [600, 527, 720, 640]

    for key in frame_ann_dict.keys():

        frameid = key - start_frame

        if frameid>=len(frames):break

        print('key ', key, ' frameid ', frameid)
        frame_ann_dict[key] = filter_anns(frame_ann_dict[key])

        # crops, crop_anns, crop_labels = pad_image(frames[frameid], frame_ann_dict[key], debug=debug)
        #if len(frame_ann_dict[key]) == 0: continue

        crop, crop_anns, crop_labels = pad_roi(frames[frameid], frame_ann_dict[key], roi, debug=debug)

        if crop_anns is None:continue

        data += crop
        bboxes += [crop_anns]
        labels += [crop_labels]

        if 0:
            img = utils.draw_bbox(frames[frameid], frame_ann_dict[key])
            img = cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 3)
            cv2.imshow('img ', img)
            cv2.waitKey(-1)
        
    return data, bboxes, labels

def save_data(data, anns, save_dir):

    imgdir_name = 'images_' + str(video_id) + '/'
    save_img_dir = os.path.join(save_dir, imgdir_name)

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    for img_id, img in enumerate(data):
        
        img_name = str(img_id) + '.jpg'

        save_img_path = os.path.join(save_img_dir, img_name)
        print('save_img_path ', save_img_path)

        cv2.imwrite(save_img_path, img)
        img_id += 1

    csv_path = os.path.join(save_dir, str(video_id) + '.csv')
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)

        # write the data
        for img_id, ann_img in enumerate(anns):

            rel_img_path = imgdir_name + str(img_id) + '.jpg'
            for ann in ann_img:
                writer.writerow([rel_img_path] + ann)


if __name__ == "__main__":

    video_id = 2067844

    ann_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/csv_files/'
    video_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/' + str(video_id) + '.mp4'
    save_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train_rcnn/data/'
    
    data, bboxes, labels = load_data(ann_dir, video_dir, debug=0)
    print(len(data), len(bboxes), len(labels))
    save_data(data, bboxes, save_dir)
    