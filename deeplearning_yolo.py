import cv2
import numpy as np
import os 
import pytesseract as pt

def get_detections(img, net):
    # convert image to yolo format
    image = img.copy()
    row, col, d = image.shape
    
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    
    # cv2.namedWindow('test image', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('test image', input_image)
    # cv2.waitKey()
    # cv2.destroyWindows()
    
    # get predictions from yolo model
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB = True, crop = False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_suppression(input_image, detections):
    # filter detectins based on confidence and probablity score
    # center_x, center_y, y, w, h, conf, proba
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT
    
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]     # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
    
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy - 0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left, top, width, height])
    
                confidences.append(confidence)
                boxes.append(box)
    
    # clean 
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist() 
    
    # nms
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.05, 0.45).flatten()
    return boxes_np, confidences_np, index

def drawings(image, boxes_np, confidences_np, index):
    # draw bounding box
    for ind in index:
        x,y,w,h= boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
    
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x,y-30), (x+w, y), (255, 0, 255), -1)
        
        cv2.putText(image, conf_text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    return image

# prediction
def yolo_predictions(img, net):
    ## step 1 detections
    input_image, detections = get_detections(img, net)
    ## step 2 nms
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    ## step 3 drawing
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img

# test
# setting
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# load yolo model
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


img = cv2.imread('./images/N33.jpeg')
results = yolo_predictions(img, net)


cv2.namedWindow('test image', cv2.WINDOW_KEEPRATIO)
cv2.imshow('test image', results)
cv2.waitKey(0)
cv2.destroyWindows()
