import cv2
import sys
from random import randint
import pandas as pd
import sys
df=pd.DataFrame()
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[2]
    df2=pd.read_csv("/home/balaji/Documents/code/RSL/Fish/bucket_detection/annotate/only_fish_116.csv")
    df2=df2[df2.Species!="VERTEBRATES, UNCLASSIFIED"]
    intervals=[]
    bboxs=[]
    for idx,row in df2.iterrows():
        intervals.append(row["frame"])
        bboxs.append((row["x"],row["y"],row["width"],row["height"]))
    # print(len(intervals))
    # sys.exit()
    def createTrackerByName(trackerType):
            # Create a tracker based on tracker name
        if trackerType == trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
           
        return tracker

    # Read video
    cap = cv2.VideoCapture("/home/balaji/Documents/code/RSL/Fish/videos/2068016.mp4")
    multiTracker = cv2.legacy.MultiTracker_create()


    # Exit if video not opened.
    if not cap.isOpened():
        sys.exit()

    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        sys.exit()
    
    # Define an initial bounding box
    bbox1=(643.4163701067622,244.93670886075984,52.583629893238395,66.06329113924049)
    bbox2=(684.4128113879002,234.68354430379776,55.58718861209958,65.31645569620255)
  
    bboxes=[bbox1,bbox2]
    colors=[]
  
  
    frame_c=0
    xs=[]
    img_array=[]
    ys=[]
    ws=[]
    hs=[]
    frame_ids=[]
    video_ids=[]
    trackers=[]
    bucket_counter=0
    while bucket_counter<len(bboxs)-1:
        
        print(frame_c)
        success, frame = cap.read()
        if not success or bucket_counter>=len(bboxs):
            break

        if(frame_c<int(intervals[bucket_counter])):
            frame_c+=1
            continue
        elif(frame_c==intervals[bucket_counter]):
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print(bucket_counter)
            multiTracker = cv2.legacy.MultiTracker_create()
            ok = multiTracker.add(cv2.legacy.TrackerCSRT_create(),frame, bboxs[bucket_counter])
            trackers.append(multiTracker)
            bucket_counter+=1
            frame_c+=1
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
    frame_c=0
    cap = cv2.VideoCapture("/home/balaji/Documents/code/RSL/Fish/videos/2068116.mp4")
    bucket_counter=1
    while bucket_counter<14:
        print(frame_c,bucket_counter)

        
        success, frame = cap.read()
        if(frame is not None):
            height,width,layers=frame.shape

        if not success:
            break
        if(frame_c<(intervals[bucket_counter-1]+intervals[bucket_counter])//2 and frame_c>intervals[bucket_counter-1]):
            success,bboxs=trackers[bucket_counter-1].update(frame)
        # get updated location of objects in subsequent frames

        # draw tracked objects
            for i, newbox in enumerate(bboxs):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[bucket_counter], 2, 1)
                x,y,w,h=(int(newbox[0]), int(newbox[1]),int(newbox[2]),int(newbox[3]))
                frame_id=frame_c
                video_id="2068116"
                xs.append(x)
                ys.append(y)
                ws.append(w)
                hs.append(h)
                frame_ids.append(frame_id)   
                video_ids.append(video_id)
        if(frame_c<intervals[bucket_counter]and frame_c>intervals[bucket_counter-1]):  
            img_array.append(frame)
        if(frame_c==intervals[bucket_counter]):
            bucket_counter+=1       

        # show frame
        # cv2.imshow('MultiTracker', frame)
        frame_c+=1

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
    size=(width,height)
    out = cv2.VideoWriter('2068116_tracked.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    df["x"]=xs
    df["y"]=ys
    df["w"]=ws
    df["h"]=hs
    df["frame_id"]=frame_ids
    df["video_id"]=video_ids
    df.to_csv("annotations_tracker_116.csv")