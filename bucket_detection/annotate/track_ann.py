from ast import AnnAssign
import cv2
import sys
from random import randint
import pandas as pd
import sys
import numpy as np
import copy

def calc_area(row, limit=140):

    w, h = row["width"],row["height"]
    if w<limit and h<limit:
        return True

    return False

def load_csv_anns(input_csv_file):

    dataframe = pd.read_csv(input_csv_file)
    fdict, act_dict = {}, {}

    for idx, row in dataframe.iterrows():

        type_ = row["Category"]
        if type_ != "Baskets": continue

        frameid = row["frame"]
        bbox = [row["x"],row["y"],row["width"],row["height"]]
        fill = row["Fill Level"]

        if row["Species"] == "VERTEBRATES, UNCLASSIFIED":
            
            if frameid not in fdict:
                fdict[frameid] = [bbox + [fill]]
            else:
                fdict[frameid].append(bbox + [fill])

        elif calc_area(row):
            if frameid not in fdict:
                act_dict[frameid] = bbox + [fill]

    return fdict, act_dict

def populate_frames(cap, sfid, annfid, num_track_frames):

    prev_frames, next_frames = [], []
    eid = annfid + num_track_frames//2

    for fdix in range(sfid, eid+1):
        ret, frame = cap.read()
        if ret is False: break

        if fdix <= annfid:
            prev_frames.append(frame)
        if fdix >= annfid:
            next_frames.append(frame)

    prev_frames.reverse()
    
    return cap, prev_frames, next_frames

def visualize_anns(anns, prev_frames_org, next_frames_org):
    
    prev_frames = copy.deepcopy(prev_frames_org[1:])
    prev_frames.reverse()

    frames = prev_frames + next_frames_org
    ann_dict = {}
    minfid = sys.maxsize
    for ann in anns:
        fid, bbox = ann[0], ann[1:]
        minfid = min(minfid, fid)
        if fid not in ann_dict:
            ann_dict[fid] = [bbox]
        else:
            ann_dict[fid].append(bbox)
    
    for key in sorted(ann_dict.keys()):
        ann = ann_dict[key][1:]
        frame = frames[key - minfid]
        print(key, key - minfid)
        vis_frame = draw_bboxes(frame, ann)
        cv2.imshow('vis_frame ', vis_frame)
        cv2.waitKey(-1)

def track_bboxes(cap, sfid, annfid, bboxes, num_track_frames, debug=1):

    cap, prev_frames, cur_frames = populate_frames(cap, sfid, annfid, num_track_frames)

    if 0:
        print('*** show prev frame ***')
        for prev_frame in prev_frames:
            cv2.imshow('prev frame ', prev_frame)
            cv2.waitKey(-1)
        
        print('*** show cur frame ***')
        for cur_frame in cur_frames:
            cv2.imshow('cur frame ', cur_frame)
            cv2.waitKey(-1)

    num_boxes = len(bboxes)
    trackers = [0] * num_boxes
    annframe = prev_frames[0]

    all_anns = []
    for i in range(num_boxes):
        trackers[i] = cv2.legacy.TrackerCSRT_create()
        all_anns += [[annfid] + bboxes[i]]

    for j in range(num_boxes):

        trackers[j].init(annframe, bboxes[j][:-1])

        for i in range(1, len(prev_frames)):
            (success, box) = trackers[j].update(prev_frames[i])
            if success:
                cfid = annfid - i
                ann = [[cfid] + list(box) + [bboxes[j][-1]]]
                all_anns += ann

        trackers[j].init(annframe, bboxes[j][:-1])

        for i in range(1, len(cur_frames)):
            (success, box) = trackers[j].update(cur_frames[i])
            if success:
                cfid = annfid + i
                ann = [[cfid] + list(box) + [bboxes[j][-1]]]
                all_anns += ann

    if debug:
        visualize_anns(all_anns, prev_frames, cur_frames)

    return cap, all_anns

def get_tracker_anns(video_path, ann_dict, num_track_frames, debug=1):

    cap = cv2.VideoCapture(video_path)
    frameid = 0

    global_anns = []
    annfids = [key for key in ann_dict.keys()]
    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid += 1

        if frameid + num_track_frames//2 in annfids:
            annid = frameid + num_track_frames//2
            bboxes = ann_dict[annid]
            print('frameid ', frameid, ' annid ', annid)
            cap, anns = track_bboxes(cap, frameid, annid, bboxes, num_track_frames, debug=debug)
            global_anns += anns

    return global_anns

def track_bbox_act(video_path, ann_dict, num_track_frames, debug=1):

    cap = cv2.VideoCapture(video_path)
    frameid = 0
    
    sfid = None
    anns = []

    while True:

        ret, frame = cap.read()
        if ret is False: break
        frameid += 1

        ann = None

        if frameid in ann_dict.keys():
            ann = ann_dict[frameid]
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, ann[:-1])
            sfid = frameid
        elif sfid is not None and (frameid - sfid) < num_track_frames:
            (success, box) = tracker.update(frame)
            if success:
                ann = list(box) + [ann_dict[sfid][-1]]
        
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

def merge_anns(static_anns, act_anns):

    static_anns = np.array(static_anns)
    act_anns = np.array(act_anns)

    print('static_anns ', static_anns.shape)
    print('act_anns ', act_anns.shape)

    idxs = np.nonzero(np.in1d(act_anns[:, 0], static_anns[:, 0]))
    print('idxs ', idxs)

    return static_anns.tolist() + act_anns[idxs].tolist()

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[2]
    videoid = 2068112
    input_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/Harmony_12/' + str(videoid) + '.csv'
    video_path = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/' + str(videoid) + '.mp4'
    save_csv_file = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/annotate/pp_' + str(videoid) + '.csv'
    num_track = 5#60
    num_track_act = 600#30
    ann_dict, act_dict = load_csv_anns(input_csv_file)

    print('bckt frames ', len(ann_dict), ann_dict.keys())
    
    #static_anns = get_tracker_anns(video_path, ann_dict, num_track, debug=1)
    act_anns = track_bbox_act(video_path, act_dict, num_track_act, debug=1)
    tot_anns = merge_anns(static_anns, act_anns)
    print(len(static_anns), len(act_anns), len(tot_anns))
    save_ann_csv(tot_anns, save_csv_file)

    # add_frame_nums(video_path)