# import the necessary packages
import numpy as np
# import argparse
import cv2
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())
# load the image
path = './images/img11.jpeg'
img = cv2.imread(path)

# define the list of boundaries
red_boudary =   [[0, 0, 100], [120, 120, 255]]      # red
blue_boundary =  [[100, 0, 0], [255, 120, 120]]     # Blue
green_boundary = [[0, 100, 0], [120, 255, 120]]     # Green
orange_boundary = [[0, 50, 100], [100, 180, 255]]   # Orange
black_boundary =  [[0, 0, 0], [50, 50, 50]]         # Black

red_lower = np.array(red_boudary[0], np.uint8)
red_upper = np.array(red_boudary[1], np.uint8)
red_mask = cv2.inRange(img, red_lower, red_upper)

blue_lower = np.array(blue_boundary[0], np.uint8)
blue_upper = np.array(blue_boundary[1], np.uint8)
blue_mask = cv2.inRange(img, blue_lower, blue_upper)

green_lower = np.array(green_boundary[0], np.uint8)
green_upper = np.array(green_boundary[1], np.uint8)
green_mask = cv2.inRange(img, green_lower, green_upper)

orange_lower = np.array(orange_boundary[0], np.uint8)
orange_upper = np.array(orange_boundary[1], np.uint8)
orange_mask = cv2.inRange(img, orange_lower, orange_upper)

black_lower = np.array(black_boundary[0], np.uint8)
black_upper = np.array(black_boundary[1], np.uint8)
black_mask = cv2.inRange(img, black_lower, black_upper)

# to detect only that particulat color
kernal = np.ones((5, 5), np.uint8)
# red color
red_mask = cv2.dilate(red_mask, kernal)
res_red = cv2.bitwise_and(img, img, mask=red_mask)

# blue color
blue_mask = cv2.dilate(blue_mask, kernal)
res_blue = cv2.bitwise_and(img, img, mask=blue_mask)

# green color
green_mask = cv2.dilate(green_mask, kernal)
res_green = cv2.bitwise_and(img, img, mask=green_mask)

# orange color
orange_mask = cv2.dilate(orange_mask, kernal)
res_orange = cv2.bitwise_and(img, img, mask=orange_mask)

# black color
black_mask = cv2.dilate(black_mask, kernal)
res_black = cv2.bitwise_and(img, img, mask=black_mask)

# creating counter to track red color
counters, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        print("Red color detected")
        break
        
# creating counter to track blue color
counters, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        print("Blue color detected")
        break
# creating counter to track blue color

counters, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        print("green color detected")
        break
        # creating counter to track orange color
counters, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        print("orange color detected")
        break
        # black counter to track black color
counters, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(counters):
    area = cv2.contourArea(contour)
    if(area > 300):
        print("black color detected")
        break