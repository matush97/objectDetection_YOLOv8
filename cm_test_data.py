# Importing the libraries
from ultralytics import YOLO
import numpy as np

from function import get_photos_folder, calculation_number_hand, plot_confusion_matrix

# Constants
path = "runs/detect/train/weights/best.pt"
directory = "test_photo/test"

not_hold = "not_hold"
one_hand_hold = "one_hand_hold"
two_hand_hold = "two_hand_hold"

confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

# Download the model
model = YOLO(path)
class_indices = model.model.names

# Get photos from folders
photos_not_hold = get_photos_folder(directory, not_hold)
photos_one_hand_hold = get_photos_folder(directory, one_hand_hold)
photos_two_hand_hold = get_photos_folder(directory, two_hand_hold)
photos_files = [photos_not_hold, photos_one_hand_hold, photos_two_hand_hold]

# Iterate through all photos and find prediction for classes - not_hold, one_hand_hold, two_hand_hold
array_classes = [not_hold, one_hand_hold, two_hand_hold]
index = 0

for photos in photos_files:
    for photo in photos:
        photo_path = directory + "/" + array_classes[index] + "/" + photo

        # Predictions from input photo
        predictions = model(photo_path)

        # List bounding boxes
        bounding_boxes = predictions[0].boxes.numpy().boxes

        # Calculation number of hands on steering wheel
        count_hand_on_wheel = calculation_number_hand(bounding_boxes)

        prediction = not_hold
        if count_hand_on_wheel == 1:
            prediction = one_hand_hold
        elif count_hand_on_wheel == 2:
            prediction = two_hand_hold

        # Increase number for class in cm
        predicted_class_index = array_classes.index(prediction)

        confusion_matrix[index, predicted_class_index] += 1

    index += 1

# Creation of confusion matrix
print(confusion_matrix)
plot_confusion_matrix(cm=confusion_matrix, classes=array_classes, title='Confusion Matrix')

