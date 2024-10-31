import cv2
import numpy as np

# Load image
image = cv2.imread('/home/samuel/Downloads/ethiopia2021.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define boundaries
red_boundary_1 = np.array([0, 120, 70]), np.array([10, 255, 255])
red_boundary_2 = np.array([170, 120, 70]), np.array([180, 255, 255])
blue_boundary = np.array([100, 150, 0]), np.array([140, 255, 255])
green_boundary = np.array([40, 70, 70]), np.array([80, 255, 255])
orange_boundary = np.array([10, 100, 100]), np.array([25, 255, 255])
black_boundary = np.array([0, 0, 0]), np.array([180, 255, 50])

# Create masks for each color
mask_red_1 = cv2.inRange(hsv_image, *red_boundary_1)
mask_red_2 = cv2.inRange(hsv_image, *red_boundary_2)
mask_red = mask_red_1 | mask_red_2
mask_blue = cv2.inRange(hsv_image, *blue_boundary)
mask_green = cv2.inRange(hsv_image, *green_boundary)
mask_orange = cv2.inRange(hsv_image, *orange_boundary)
mask_black = cv2.inRange(hsv_image, *black_boundary)

# Combine results or apply each mask individually as needed
red_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_red)
blue_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_blue)
green_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_green)
orange_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_orange)
black_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_black)


# to detect only that particulat color
kernal = np.ones((5, 5), np.uint8)
# red color
red_mask = cv2.dilate(red_mask, kernal)
res_red = cv2.bitwise_and(hsv_image, hsv_image, mask=red_mask)

# blue color
blue_mask = cv2.dilate(blue_mask, kernal)
res_blue = cv2.bitwise_and(hsv_image, hsv_image, mask=blue_mask)

# green color
green_mask = cv2.dilate(green_mask, kernal)
res_green = cv2.bitwise_and(hsv_image, hsv_image, mask=green_mask)

# orange color
orange_mask = cv2.dilate(orange_mask, kernal)
res_orange = cv2.bitwise_and(hsv_image, hsv_image, mask=orange_mask)

# black color
black_mask = cv2.dilate(black_mask, kernal)
res_black = cv2.bitwise_and(hsv_image, hsv_image, mask=black_mask)


# creating counter to track red color
counters, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        # print("Red color detected")
        # break
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Red color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imwrite('image_red_detected.jpg', hsv_image)
# creating counter to track blue color
counters, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        # print("Blue color detected")
        # break
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, "blue color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imwrite('image_blue_detected.jpg', img )