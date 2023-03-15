# Importing the libraries
from ultralytics import YOLO
import cv2
import numpy as np
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

from shapely.geometry import box

path = "runs/detect/train/weights/best.pt"
model = YOLO(path)
label_map = model.model.names

# img_path = "datasets/test/images/IMG_20230219_143401_jpg.rf.22d766e08fa08b90d1925866f0e0b54d.jpg"
# img_path = "datasets/test/images/IMG_20221228_151629_3_jpg.rf.f31f716db9746b889cf79cccd6271acf.jpg"
img_path = "datasets/test/images/IMG_20230107_133233_jpg.rf.550396488c67d35fadc6224b47c16d3d.jpg"

predictions = model(img_path)

# list bounding boxes
bounding_boxes = predictions[0].boxes.numpy().boxes

length_bounding_boxes = len(bounding_boxes)
x = 0
for i in range(0, length_bounding_boxes):
    first_array = bounding_boxes[i]
    for j in range(x, length_bounding_boxes):
        second_array = bounding_boxes[j]
        # compare_arrays =
        if np.array_equal(first_array, second_array):
            continue

        # area
        rectangle1 = box(first_array[0], first_array[1], first_array[2], first_array[3])
        rectangle2 = box(second_array[0], second_array[1], second_array[2], second_array[3])

        intersection = rectangle1.intersection(rectangle2).area / rectangle1.union(rectangle2).area

        print(intersection)
    x += 1

# print(bounding_boxes)

imageShow = cv2.imread(img_path)

# vykreslenie bounding boxov
for bounding_box in bounding_boxes:
    x0 = bounding_box[0]
    x1 = bounding_box[2]
    y0 = bounding_box[1]
    y1 = bounding_box[3]

    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))
    if bounding_box[5] == 1:
        cv2.rectangle(imageShow, start_point, end_point, color=(0, 255, 0), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        imageInfo = '%s (%.2f%%)' % ("steering_wheel", bounding_box[4])
        cv2.putText(imageShow, imageInfo, start_point, font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    elif bounding_box[5] == 0:
        cv2.rectangle(imageShow, start_point, end_point, color=(0, 0, 255), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        imageInfo = '%s (%.2f%%)' % ("hand", bounding_box[4])
        cv2.putText(imageShow, imageInfo, start_point, font, 1, (0, 0, 255), 1, cv2.LINE_AA)


# TODO zistovanie prekrytia ruk a volantu


cv2.imwrite("example_with_bounding_boxes.jpg", imageShow)
cv2.imshow('Photo', imageShow)
cv2.waitKey(0)
