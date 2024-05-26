from ultralytics import YOLO

model = YOLO('yolov9e-seg.pt')
dataset_yaml = 'rooms_dataset/data.yaml'

results = model.train(
    data=dataset_yaml,
    epochs=40,
    imgsz=640,
    batch=4,
    degrees=90,
    translate=0.01,
    perspective=0.0001,
)
