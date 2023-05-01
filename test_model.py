# Importing the libraries
from ultralytics import YOLO
import cv2

from function import calculation_number_hand

# Constants
img_path = "test_photo/image1.jpg"
path = "runs/detect/train/weights/best.pt"

class_position = 5
font = cv2.FONT_HERSHEY_SIMPLEX

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

    object_color = class_color.get(bounding_box[class_position])
    object_name = class_indices.get(bounding_box[class_position])

    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))

    cv2.rectangle(imageShow, start_point, end_point, color=object_color, thickness=2)

    imageInfo = '%s (%.2f%%)' % (object_name, bounding_box[4])
    cv2.putText(imageShow, imageInfo, start_point, font, 2, object_color, 4, cv2.LINE_AA)

# Calculation number of hands on steering wheel
count_hand_on_wheel = calculation_number_hand(bounding_boxes)


output = "not_hold"
if count_hand_on_wheel == 1:
    output = "one_hand_hold"
elif count_hand_on_wheel == 2:
    output = "two_hand_hold"

print(output)

# Add title
cv2.putText(imageShow, output, (50, 150), font, 2.5, (0, 0, 255), 4, cv2.LINE_AA)

# Resize picture
scale_percent = 40
width = int(imageShow.shape[1] * scale_percent / 100)
height = int(imageShow.shape[0] * scale_percent / 100)
dsize = (width, height)
imageShow = cv2.resize(imageShow, dsize)

# Save an image to storage device
cv2.imwrite("example_with_bounding_boxes.jpg", imageShow)

cv2.imshow('Photo', imageShow)
cv2.waitKey(0)
