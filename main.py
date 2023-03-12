from ultralytics import YOLO

# from IPython.display import display, Image

# Download model and dataset
model = YOLO('yolov8n.pt')
# results = model.predict(
#     source='./volant.jpg',
#     conf=0.25
# )
#
# print(results)
#
from roboflow import Roboflow
# version 1
# rf = Roboflow(api_key="PJxJ3H7MF1K3bJprv5KA")
# project = rf.workspace("slovak-technical-university").project("steering-wheels-01")
# dataset = project.version(1).download("yolov8")

# version 2
# rf = Roboflow(api_key="PJxJ3H7MF1K3bJprv5KA")
# project = rf.workspace("slovak-technical-university").project("steering-wheels-01")
# dataset = project.version(2).download("yolov8")

# Train model
path = "data.yaml"

# Use the model
model.train(data=path, epochs=25, imgsz=640)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("volant.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format

# yolo task=detect mode=train  model=yolov8n.pt  data={dataset.location}/data.yaml  epochs=10 imgsz=640