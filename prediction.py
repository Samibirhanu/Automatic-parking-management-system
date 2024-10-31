import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load model

model = tf.keras.models.load_model('./object_detection.keras')
print('model loaded sucessfully')

# path = './images/N207.jpeg'
# image = load_img(path)
# image = np.array(image, dtype = np.uint8)   # 8 bit array (0, 255)
# image1 = load_img(path, target_size = (224, 224))
# image_arr_224 = img_to_array(image1)/255.0      # convert into array and get the normalized output

# # print size of the original image
# h, w, d = image.shape
# print("height of the image =", h)
# print("width of the image =", w)

# # plt.imshow(image)
# # plt.show()

# print(image_arr_224.shape)

# test_arr = image_arr_224.reshape(1, 224, 224, 3)
# test_arr.shape

# # make prediction

# coords = model.predict(test_arr)
# # print(coords)
# print(len(coords))

# create pipline
# path = '/content/drive/MyDrive/train-object-detection/images_labeled/N230.jpeg'
def object_detection(path):
  # read image
  image = load_img(path)
  image = np.array(image, dtype = np.uint8)   # 8 bit array (0, 255)
  image1 = load_img(path, target_size = (224, 224))

  # data preprocessing
  image_arr_224 = img_to_array(image1)/255.0      # convert into array and get the normalized output
  h, w, d = image.shape
  test_arr = image_arr_224.reshape(1, 224, 224, 3)

  # make prediction
  coords = model.predict(test_arr)

  # denormalized
  denorm = np.array([w, w, h, h])
  coords = coords * denorm
  coords = coords.astype(np.int32)

  # draw bounding on top the image
  xmin, xmax, ymin, ymax = coords[0]
  pt1 = (xmin, ymin)
  pt2 = (xmax, ymax)
  pt1, pt2
  cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
  return image, coords

# path = './images/N233.jpeg'
# image, cods = object_detection(path)
# plt.figure()
# plt.imshow(image)
# plt.show()



# Load the image using OpenCV
#/home/samuel/Downloads/sample_ethiopian.jpg download location
path = '/home/samuel/Downloads/sample_ethiopian.jpg'
image, cods = object_detection(path)

# Convert the image from BGR to RGB (since OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using OpenCV
output_path = '/home/samuel/Downloads/inceptionrsnet_detection/sample_ethiopian.jpg'
cv2.imwrite(output_path, image_rgb)
cv2.imshow('Detected Objects', image_rgb)

# Wait for a key press and close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()

# import pytesseract as pt

# img = load_img(path)
# img = np.array(img)
# xmin, xmax, ymin, ymax = cods[0]
# roi = img[ymin:ymax, xmin:xmax]
# plt.imshow(roi)
# plt.show()


# extract text from image
# extracted_plate = pt.image_to_string(roi)
# print(extracted_plate)

