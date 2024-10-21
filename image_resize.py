import cv2

# Load the image
image = cv2.imread('/home/samuel/Downloads/stella_car_1.jpg')

# Specify the new size (width, height)
new_size = (300, 200)

# Resize the image
resized_image = cv2.resize(image, new_size)

# Show the resized image
output_path = '/home/samuel/Downloads/stella_car_1_resized.jpg'
cv2.imwrite(output_path, resized_image)
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
