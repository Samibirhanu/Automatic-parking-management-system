import cv2
import numpy as np
import os 
import pytesseract as pt

# Function to get detections from an image using YOLO model
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # YOLO input formatting
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

# Function to apply Non-Maximum Suppression (NMS) to filter overlapping boxes
def non_maximum_suppression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.05, 0.45).flatten()
    
    return boxes_np, confidences_np, index

# settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# load yolo model
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to detect objects using YOLO and return bounding box coordinates
def yolo_object_detection(image_path, filename, net):
    # Read the image
    img = cv2.imread(image_path)
    row, col, _ = img.shape
    input_image, detections = get_detections(img, net)

    # Non-maximum suppression to filter out overlapping boxes
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)

    if len(index) > 0:
        # Get the coordinates for the first detected plate (since index may contain multiple)
        ind = index[0]
        x, y, w, h = boxes_np[ind]

        # Draw bounding box on the original image
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        cv2.imwrite(f'./static/upload/predict/{filename}', img)

        # Return bounding box coordinates
        coords = [x, x + w, y, y + h]
        return coords
    else:
        return None  # No detections found
    

def yolo_OCR(image_path, filename):
    # Perform object detection to get bounding box coordinates
    coords = yolo_object_detection(image_path, filename, net)
    
    if coords is not None:
        # Extract ROI based on the detected coordinates
        img = cv2.imread(image_path)
        xmin, xmax, ymin, ymax = coords
        roi = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(f'./static/upload/roi/{filename}', roi)

        # Perform OCR on the cropped region
        text = pt.image_to_string(roi)
        print(text)
        return text
    else:
        return "Unable To Recognize"  # No text if no detection
