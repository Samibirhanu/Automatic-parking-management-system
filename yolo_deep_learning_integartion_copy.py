import cv2
import numpy as np
import os 
import pytesseract as pt
import easyocr

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
    indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.05, 0.45)

    # Check if 'indices' is a non-empty list or array, then flatten if necessary
    if isinstance(indices, tuple) and len(indices) > 0:
        indices = indices[0].flatten()  # Access the first element and flatten
    elif isinstance(indices, list) and len(indices) > 0:
        indices = np.array(indices).flatten()  # Convert to array and flatten if it's a list

# 'indices' will now be a flattened array if boxes were found, or empty otherwise.


    
    return boxes_np, confidences_np, indices

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
    
reader = easyocr.Reader(['en'])
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
        perform_ocr = reader.readtext(roi)
        text = " ".join([result[1] for result in perform_ocr])
        print(text)
        return text, coords
    else:
        return "Unable To Recognize"  # No text if no detection


def plate_number_sequence(text):
    index = 0
    while index < len(text) - 1:
        # Check if the current character is a letter followed by a number
        if text[index].isalpha() and text[index + 1].isdigit():
            # Try to get a sequence of five characters starting from this point
            sequence = text[index:index + 6]
            # Check if we have exactly one letter followed by four numbers
            if len(sequence) == 6 and sequence[1:].isdigit():
                return sequence
        # Move to the next character
        index += 1
    return "No matching sequence found"