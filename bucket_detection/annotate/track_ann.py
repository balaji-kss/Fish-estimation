import cv2
import sys
from random import randint
import pandas as pd
import sys
import numpy as np

def calc_area(row, limit=140):

    w, h = row["width"],row["height"]
    if w<limit and h<limit:
        return True

    return False

def load_csv_anns(input_csv_file):

    dataframe = pd.read_csv(input_csv_file)
    #dataframe = dataframe[dataframe.Species!="VERTEBRATES, UNCLASSIFIED"]
    m_intervals, m_bboxes, m_fill_ests = [], [], []
    s_intervals, s_bboxes, s_fill_ests = [], [], []

    for idx, row in dataframe.iterrows():

        type_ = row["Category"]
        if type_ != "Baskets": continue
        
        if row["Species"] == "VERTEBRATES, UNCLASSIFIED":
            s_intervals.append(row["frame"])
            s_bboxes.append([row["x"],row["y"],row["width"],row["height"]])
            s_fill_ests.append(row["Fill Level"])
        
        elif calc_area(row):
            m_intervals.append(row["frame"])
            m_bboxes.append([row["x"],row["y"],row["width"],row["height"]])
            m_fill_ests.append(row["Fill Level"])

    moving_anns = [m_intervals, m_bboxes, m_fill_ests]
    static_anns = [s_intervals, s_bboxes, s_fill_ests]

    return moving_anns, static_anns 

def get_tracker_anns(video_path, anns, num_track_frames, debug=1):

    intervals, bboxes, fill_ests = anns[0], anns[1], anns[2]
    cap = cv2.VideoCapture(video_path)
    frameid = 0
    
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
            ann = bboxes[index]  + [fill_ests[index]]
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, bboxes[index])
            trackers[index] = tracker
            cur_track_idx = index
        elif cur_track_idx is not None and (frameid - intervals[cur_track_idx]) < num_track_frames:
            (success, box) = trackers[cur_track_idx].update(frame)
            if success:
                ann = list(box) + [fill_ests[cur_track_idx]]
        
        if ann is None: continue

        if debug:
            print(frameid)
            vis_frame = draw_bboxes(frame, [ann])
            cv2.imshow('vis_frame ', vis_frame)
            cv2.waitKey(-1)

        anns.append([frameid] + ann)
    
    return anns

def dict_frame_anns(anns):

    intervals, bboxes, fill_ests = anns[0], anns[1], anns[2]

    frame_dict = {}

    for interval, bbox, fill in zip(intervals, bboxes, fill_ests):
        if interval not in frame_dict:
            frame_dict[interval] = [bbox + [fill]]
        else:
            frame_dict[interval].append(bbox + [fill])
    
    return frame_dict

def duplicate_anns(ann, start_fid, num_track_frames):
    
    ann = np.array(ann).reshape((-1, 5))
    num_bboxes = ann.shape[0]
    dup_anns = []

    for num_frame in range(start_fid, start_fid + num_track_frames):
        fids = [num_frame] * num_bboxes
        fids = np.array(fids).reshape((-1, 1))
        concat_dup_ann = np.concatenate((fids, ann), axis=1)
        concat_dup_ann = concat_dup_ann.tolist()
        dup_anns += concat_dup_ann

    return dup_anns

def get_static_anns(video_path, anns, num_track_frames, debug=0):

    cap = cv2.VideoCapture(video_path)
    frameid = 0
    fdict = dict_frame_anns(anns)
    preprocess_anns = []

    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid += 1

        if frameid in fdict:
            ann = fdict[frameid]
            dup_ann = duplicate_anns(ann, frameid, num_track_frames)
            preprocess_anns += dup_ann

            if debug:
                vis_frame = draw_bboxes(frame, ann)
                cv2.imshow('vis_frame ', vis_frame)
                cv2.waitKey(-1)

    return preprocess_anns

def draw_bboxes(org_image, anns, color=((255, 0, 0))):

    image = np.copy(org_image)

    for ann in anns:
        startX, startY, w, h = ann[:-1]
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
    df["fill"] = anns_np[:, 5].tolist()
    df["video_id"] = [2068116] * len(df["x"])

    df.to_csv(save_csv_file)

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[2]
    
    input_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/Harmony_12/2067840.csv'
    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/2067840.mp4'
    save_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/pp_2067840.csv'
    num_track_static, num_track_move = 50, 30
    moving_bckt_anns, static_bckt_anns = load_csv_anns(input_csv_file)

    print('moving bckt frames ', moving_bckt_anns[0])
    print('static bckt frames ', static_bckt_anns[0])

    move_anns = get_tracker_anns(video_path, moving_bckt_anns, num_track_move, debug=1)
    
    stat_anns = get_static_anns(video_path, static_bckt_anns, num_track_static, debug=1)

    tot_anns = move_anns + stat_anns
    save_ann_csv(tot_anns, save_csv_file)

    # add_frame_nums(video_path)