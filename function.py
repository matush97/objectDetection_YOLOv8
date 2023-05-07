# Importing the libraries
import os
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Functions
def get_photos_folder(directory, folder_name):
    for item in os.listdir(directory):
        if item != folder_name:
            continue

        return os.listdir(directory + "/" + folder_name)


def calculation_number_hand(bounding_boxes, class_position=5):
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
    return count_prediction


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("classes_confusion_matrix.png")
