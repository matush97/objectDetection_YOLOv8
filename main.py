# Import the libraries
from ultralytics import YOLO

# Constants
path = "data.yaml"

# Download model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data=path, epochs=10, imgsz=640)  # train the model
model.val()  # evaluate model performance on the validation set
