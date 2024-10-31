import cv2
import numpy as np

def plate_color_detection(image_path):
    
  # Load the image
  img = cv2.imread(image_path)

  # Convert the image from BGR to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Define color boundaries for red, blue, and green in HSV
  red_lower = np.array([0, 100, 100])
  red_upper = np.array([10, 255, 255])

  blue_lower = np.array([100, 100, 100])
  blue_upper = np.array([140, 255, 255])

  green_lower = np.array([40, 50, 50])
  green_upper = np.array([80, 255, 255])

  # Create masks for each color
  mask_red = cv2.inRange(hsv, red_lower, red_upper)
  mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
  mask_green = cv2.inRange(hsv, green_lower, green_upper) 


  # Apply masks to the image
  res_red = cv2.bitwise_and(img, img, mask=mask_red)
  res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
  res_green = cv2.bitwise_and(img, img, mask=mask_green)
  # count non zero pixels
  red_count = cv2.countNonZero(mask_red)
  blue_count = cv2.countNonZero(mask_blue)
  green_count = cv2.countNonZero(mask_green)

  # Determine the majority color
  if blue_count > green_count and blue_count > red_count:
      # print("Blue is the majority color")
      majority_pixles = blue_count
      color = 'Blue'
  elif green_count > blue_count and green_count > red_count:
      # print("Green is the majority color")
      majority_pixles = green_count
      color = 'Green'
  elif red_count > blue_count and red_count > green_count:
      # print("Red is the majority color")
      majority_pixles = red_count
      color = 'Red'
  return color
  # return f'{identifier}{color}'
