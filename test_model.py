# Importing the libraries
from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import box

# Constants
img_path = "test_photo/image1.jpg"
path = "runs/detect/train/weights/best.pt"

# Download the model
model = YOLO(path)
class_indices = model.model.names

# Predictions from input photo
predictions = model(img_path)

# List bounding boxes
bounding_boxes = predictions[0].boxes.numpy().boxes

imageShow = cv2.imread(img_path)

# Colors for classes
class_color = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0)
}


# Drawing a bounding boxes - with name and percentage
for bounding_box in bounding_boxes:
    x0 = bounding_box[0]
    x1 = bounding_box[2]
    y0 = bounding_box[1]
    y1 = bounding_box[3]

    object_color = class_color.get(bounding_box[5])
    object_name = class_indices.get(bounding_box[5])

    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))

    cv2.rectangle(imageShow, start_point, end_point, color=object_color, thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    imageInfo = '%s (%.2f%%)' % (object_name, bounding_box[4])
    cv2.putText(imageShow, imageInfo, start_point, font, 2, object_color, 4, cv2.LINE_AA)

# Hand and steering wheel overlap detection
length_bounding_boxes = len(bounding_boxes)
x = 0
count_prediction = 0

# for i in range(0, length_bounding_boxes):
#     first_array = bounding_boxes[i]
#     for j in range(x, length_bounding_boxes):
#         second_array = bounding_boxes[j]
#         # compare_arrays =
#         if np.array_equal(first_array, second_array):
#             continue
#
#         # area
#         rectangle1 = box(first_array[0], first_array[1], first_array[2], first_array[3])
#         rectangle2 = box(second_array[0], second_array[1], second_array[2], second_array[3])
#
#         intersection = rectangle1.intersection(rectangle2).area / rectangle1.union(rectangle2).area
#
#         print(intersection)
#
#         if intersection > 0:
#             count_prediction += 1
#     x += 1

# Output
output = "nedrzi"
# if count_prediction == 1:
#     output = "drzi s jednou rukou"
# elif count_prediction == 2:
#     output = "drzi s dvoma rukami"

print(output)

# Save an image to storage device
# cv2.imwrite("example_with_bounding_boxes.jpg", imageShow)

# Resize picture
scale_percent = 40
width = int(imageShow.shape[1] * scale_percent / 100)
height = int(imageShow.shape[0] * scale_percent / 100)
dsize = (width, height)
imageShow = cv2.resize(imageShow, dsize)

cv2.imshow('Photo', imageShow)
cv2.waitKey(0)
