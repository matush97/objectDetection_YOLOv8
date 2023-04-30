from ultralytics import YOLO


# Download model and dataset
model = YOLO('yolov8n.pt')
# Train model
path = "data.yaml"

# Train the model
model.train(data=path, epochs=25, imgsz=224)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("volant.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format