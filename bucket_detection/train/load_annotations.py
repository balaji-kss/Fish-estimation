import numpy as np
from imutils import paths
import config
import cv2
import os 
import csv
    
def populate_frames(video_path, skip_frames, start_frame, max_num_frame=100):

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
    
def draw_bbox(org_image, anns):

    image = np.copy(org_image)
    for ann in anns:

        startX, startY, endX, endY = ann
        w, h = endX - startX, endY - startY

        image = cv2.putText(image, str(w) + ', ' + str(h), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)        
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 1)

    return image

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
            
            row = [int(ele) for ele in row]
            
            (idx, startX, startY, w, h, frame_id, video_id) = row
            endX, endY = startX + w, startY + h

            if frame_id%skip_frames!=0:continue
            
            if frame_id in frame_dict:
                frame_dict[frame_id].append([startX, startY, endX, endY])
            else:
                frame_dict[frame_id] = [[startX, startY, endX, endY]]
    
    return frame_dict

def filter_anns(anns, area_max = 120 * 120):

    filtered_anns = []

    for ann in anns:
        startX, startY, endX, endY = ann
        w, h = endX - startX, endY - startY

        if w <= 0 or h <= 0 or w * h >= area_max:continue

        filtered_anns.append([startX, startY, endX, endY])

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
            crop = draw_bbox(crop, [crop_ann])        

            cv2.imshow('img ', crop)
            cv2.waitKey(-1)

        # pp_crop = preprocess_image(crop)
        # pp_crop_anns = preprocess_anns(ann, pp_crop.shape)
        crops.append(crop)
        crop_anns.append(crop_ann)
        labels.append('bucket')

    return crops, crop_anns, labels
     
def load_data(ann_dir, video_path, debug):

    print("[INFO] loading dataset...")
    data = []
    bboxes = []
    labels = []

    skip_frames = 1
    frames = populate_frames(video_path, skip_frames, 0)
    frame_ann_dict = populate_anns(ann_dir, skip_frames)

    print('total frames: ', len(frames))
    print('total anns: ', len(frame_ann_dict))

    image_id = 0
    for frameid, key in enumerate(frame_ann_dict.keys()):

        if frameid>=len(frames):break

        # print('key ', key, ' frameid ', frameid)
        frame_ann_dict[key] = filter_anns(frame_ann_dict[key])

        crops, crop_anns, crop_labels = pad_image(frames[frameid], frame_ann_dict[key], debug=debug)

        data += crops
        bboxes += crop_anns
        labels += crop_labels

        if 0:
            img = draw_bbox(frames[frameid], frame_ann_dict[key])
            
            cv2.imshow('img ', img)
            cv2.waitKey(-1)

        

    return data, bboxes, labels

def save_data(data, anns, save_dir):

    assert len(data) == len(anns)

    for img_id, img in enumerate(data):
        
        img_name = 'images/' + str(img_id) + '.jpg'
        save_img_path = os.path.join(save_dir, img_name)
        print('save_img_path ', save_img_path)

        cv2.imwrite(save_img_path, img)
        img_id += 1

    csv_path = os.path.join(save_dir, 'bucket_anns.csv')
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)

        # write the data
        for img_id, ann in enumerate(anns):

            rel_img_path = 'images/' + str(img_id) + '.jpg'
            
            writer.writerow([rel_img_path] + ann)


if __name__ == "__main__":

    ann_dir = '/home/balaji/Documents/code/RSL/Fish/bucket_detection/annotate/csv_files/'
    video_dir = '/home/balaji/Documents/code/RSL/Fish/videos/2068016.mp4'
    save_dir = '/home/balaji/Documents/code/RSL/Fish/bucket_detection/train/data/'
    data, bboxes, labels = load_data(ann_dir, video_dir, debug=0)
    save_data(data, bboxes, save_dir)
    