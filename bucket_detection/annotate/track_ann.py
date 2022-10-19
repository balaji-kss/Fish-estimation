import cv2
import sys
from random import randint
import pandas as pd
import sys
import numpy as np

def load_csv_anns(input_csv_file):

    dataframe = pd.read_csv(input_csv_file)
    dataframe = dataframe[dataframe.Species!="VERTEBRATES, UNCLASSIFIED"]
    intervals, bboxes, fill_ests = [], [], [] 

    for idx, row in dataframe.iterrows():
        intervals.append(row["frame"])
        bboxes.append([row["x"],row["y"],row["width"],row["height"]])
        fill_ests.append(row["Fill Level"])

    return intervals, bboxes, fill_ests

def get_tracker_anns(video_path, intervals, bboxes):

    cap = cv2.VideoCapture(video_path)
    frameid = 0
    
    num_track_frames = 150
    trackers = [0] * len(intervals)
    cur_track_idx = None
    anns = []

    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid += 1

        ann = None

        if frameid in intervals:
            index = intervals.index(frameid)
            ann = bboxes[index]
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, ann)
            trackers[index] = tracker
            cur_track_idx = index
        elif cur_track_idx is not None and (frameid - intervals[cur_track_idx]) < num_track_frames:
            (success, box) = trackers[cur_track_idx].update(frame)
            if success:
                ann = list(box)
        
        if ann is None: continue
        
        anns.append([frameid] + ann)
    
    return anns

def draw_bbox(org_image, ann, color=((255, 0, 0))):

    image = np.copy(org_image)

    startX, startY, w, h = ann
    endX, endY = startX + w, startY + h
    startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
    # image = cv2.putText(image, str(w) + ', ' + str(h), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)        
    image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 1)

    return image

def add_frame_nums(input_video_path):

    out_video_path = input_video_path[:-4] + '_f.mp4'
    cap = cv2.VideoCapture(input_video_path)
    frameid = 0
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid += 1
        
        frame = cv2.putText(frame, str(frameid), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)        
        
        # cv2.imshow('frame ', frame)
        # cv2.waitKey(-1)

        out.write(frame)

    cap.release()
    out.release()

def save_ann_csv(anns, save_csv_file):

    anns_np = np.array(anns)
    df = pd.DataFrame()

    df["frame_id"] = anns_np[:, 0].tolist()
    df["x"] = anns_np[:, 1].tolist()
    df["y"] = anns_np[:, 2].tolist()
    df["w"] = anns_np[:, 3].tolist()
    df["h"] = anns_np[:, 4].tolist()
    
    df["video_id"] = [2068116] * len(df["x"])

    df.to_csv(save_csv_file)

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[2]
    
    input_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/only_fish_116.csv'
    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/videos/2068116_f.mp4'
    save_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/det_est_ann_116.csv'

    intervals, bboxes, fill_ests = load_csv_anns(input_csv_file)
    # anns = get_tracker_anns(video_path, intervals, bboxes)

    # save_ann_csv(anns, save_csv_file)

    # add_frame_nums(video_path)