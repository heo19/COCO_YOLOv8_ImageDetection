from ultralytics import YOLO

model = YOLO("yolov8n.pt")

result = model.train(data="coco.yaml", epochs=100, imgsz=640)