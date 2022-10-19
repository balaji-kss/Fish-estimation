import numpy as np
import cv2

def make_square(image):

    h, w = image.shape[:2]
    sz = max(h, w)

    pad_image = np.zeros((sz, sz, 3), dtype='uint8')
    pad_image[:h, :w] = image
    
    return pad_image

def remap_image(image):

    image *= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)

    return image

def get_bucket_roi(image, bbox, pad_range=(0.5, 0.701)):
        
    pad_min, pad_max = pad_range
    pad = np.random.uniform(pad_min, pad_max)
    H, W = image.shape[:2]

    sx, sy, ex, ey = bbox
    w, h = ex - sx, ey - sy
    cx, cy = sx + 0.5 * w, sy + 0.5 * h 
    sz = 0.5 * (w + h)

    sx, sy = cx - pad * sz, cy - pad * sz
    ex, ey = cx + pad * sz, cy + pad * sz

    sx, sy = max(0, sx), max(0, sy)
    ex, ey = min(W, ex), min(H, ey)

    sx, sy = int(sx), int(sy)
    ex, ey = int(ex), int(ey)

    crop = image[sy:ey, sx:ex]

    return make_square(crop)

def preprocess_image(image, input_res):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (input_res, input_res))
    image /= 255.0
    
    return image