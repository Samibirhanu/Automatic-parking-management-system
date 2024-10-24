import cv2
import numpy as np
import os 
import pytesseract as pt

# Create directories if not exist
os.makedirs('./static/upload/predict', exist_ok=True)
os.makedirs('./static/upload/roi', exist_ok=True)

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

def save_cropped_images(image, boxes_np, index, img_path):
    cropped_images = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        cropped_img = image[y:y+h, x:x+w]
        
        # Save full image
        save_predict_path = os.path.join('./static/upload/predict', os.path.basename(img_path))
        cv2.imwrite(save_predict_path, image)
        
        # Save cropped ROI
        roi_filename = f"roi_{ind}_{os.path.basename(img_path)}"
        save_roi_path = os.path.join('./static/upload/roi', roi_filename)
        cv2.imwrite(save_roi_path, cropped_img)
        cropped_images.append(save_roi_path)

    return cropped_images

def extract_text_from_image(cropped_image_path):
    # Read the cropped image and use pytesseract to extract text
    cropped_img = cv2.imread(cropped_image_path)
    text = pt.image_to_string(cropped_img)
    return text

def drawings(image, boxes_np, confidences_np, index, img_path):
    # draw bounding box and save cropped images
    cropped_images = save_cropped_images(image, boxes_np, index, img_path)

    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image, cropped_images

def yolo_predictions(img, net, img_path):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    result_img, cropped_images = drawings(img, boxes_np, confidences_np, index, img_path)
    
    # Extract text from each cropped image
    extracted_texts = [extract_text_from_image(roi) for roi in cropped_images]

    return result_img, extracted_texts

# settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# load yolo model
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

img_path = './images/N33.jpeg'
img = cv2.imread(img_path)
results, texts = yolo_predictions(img, net, img_path)

# Display results
# cv2.imshow('test image', results)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Print extracted texts
for i, text in enumerate(texts):
    print(f"Text from ROI {i}: {text}")

