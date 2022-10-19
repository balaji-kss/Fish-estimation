import numpy as np
import cv2

def draw_bbox(org_image, ann, color=((255, 0, 0))):

    image = np.copy(org_image)

    startX, startY, endX, endY = ann
    startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
    # image = cv2.putText(image, str(w) + ', ' + str(h), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)        
    image = cv2.rectangle(image, (startX, startY), (endX, endY), color, 1)

    return image

def remap_bbox(pred_bbox, img_shape, scale=1):

	H, W = img_shape[:2]
	pred_bbox = np.squeeze(pred_bbox)

	startX, startY, endX, endY = pred_bbox
	startX, startY, endX, endY = startX * W/scale, startY * H/scale, endX * W/scale, endY * H/scale
	startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)

	return [startX, startY, endX, endY]