from ultralytics import YOLO

path = "runs/detect/train/weights/best.pt"
model = YOLO(path)

path = "datasets/test/images/IMG_20230219_143401_jpg.rf.22d766e08fa08b90d1925866f0e0b54d.jpg"
# path = "datasets/test/images"
# path = "datasets/train/images/1_png.rf.adbf95863b747b8e2d8632cea45d890e.jpg"

results = model.predict(
    source=path,
    conf=0.25,
    save=True
)

print(results)

