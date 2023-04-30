# Download dataset from Roboflow
from roboflow import Roboflow

# version 1
# rf = Roboflow(api_key="PJxJ3H7MF1K3bJprv5KA")
# project = rf.workspace("slovak-technical-university").project("steering-wheels-01")
# dataset = project.version(1).download("yolov8")

# version 2
rf = Roboflow(api_key="PJxJ3H7MF1K3bJprv5KA")
project = rf.workspace("slovak-technical-university").project("steering-wheels-syntetic-data")
dataset = project.version(1).download("yolov8")
