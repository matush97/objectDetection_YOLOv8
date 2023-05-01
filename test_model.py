# Importing the libraries
from ultralytics import YOLO
import cv2
from shapely.geometry import box

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

# Hand and steering wheel overlap detection
length_bounding_boxes = len(bounding_boxes)
count_prediction = 0

for i in range(0, length_bounding_boxes):
    first_array = bounding_boxes[i]

    # if there is no class for hand-hold
    if first_array[class_position] != 0:
        continue

    for j in range(0, length_bounding_boxes):
        second_array = bounding_boxes[j]

        # if there is no class for steering-wheel
        if second_array[class_position] != 2:
            continue

        # area
        rectangle1 = box(first_array[0], first_array[1], first_array[2], first_array[3])
        rectangle2 = box(second_array[0], second_array[1], second_array[2], second_array[3])

        intersection = rectangle1.intersection(rectangle2).area / rectangle1.union(rectangle2).area

        if intersection > 0:
            count_prediction += 1

output = "not_hold"
if count_prediction == 1:
    output = "one_hand_hold"
elif count_prediction == 2:
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
