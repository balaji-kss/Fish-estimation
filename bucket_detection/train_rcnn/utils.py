import numpy as np
import cv2

def draw_bbox(org_image, anns):

    image = np.copy(org_image)
    for ann in anns:

        if len(ann) == 5:
            startX, startY, endX, endY, fill = ann
        elif len(ann) == 4:
            startX, startY, endX, endY = ann
            
        w, h = endX - startX, endY - startY

        # image = cv2.putText(image, str(w) + ', ' + str(h) + ', ' + str(fill), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)        
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 1)

    return image

def remap_bbox(pred_bboxes, img_shape, scale=1):

    H, W = img_shape[:2]

    remap_bboxes = []
    for pred_bbox in pred_bboxes:
        startX, startY, endX, endY = pred_bbox
        startX, startY, endX, endY = startX * W/scale, startY * H/scale, endX * W/scale, endY * H/scale
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        remap_bboxes.append([startX, startY, endX, endY])

    return remap_bboxes

def remap_image(image):

    image *= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)

    return image

def get_pad_crop(roi, pad, img_shape):

    sx, sy, ex, ey = roi
    w, h = ex - sx, ey - sy
    cx, cy =  sx + 0.5 * w, sy + 0.5 * h 
    sz = 0.5 * (w + h)
    H, W = img_shape[:2]

    sx, sy = cx - pad * sz, cy - pad * sz
    ex, ey = cx + pad * sz, cy + pad * sz

    sx, sy = max(0, sx), max(0, sy)
    ex, ey = min(W, ex), min(H, ey)

    sx, sy = int(sx), int(sy)
    ex, ey = int(ex), int(ey)

    return [sx, sy, ex, ey]

def make_square(image):

    h, w = image.shape[:2]
    sz = max(h, w)

    pad_image = np.zeros((sz, sz, 3), dtype='uint8')
    pad_image[:h, :w] = image
    
    return pad_image