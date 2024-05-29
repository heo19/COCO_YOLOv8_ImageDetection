from ultralytics import YOLO
import os
import torch
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in model.names]

cv2.namedWindow("Detecting Video", cv2.WINDOW_NORMAL)

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    results = model.predict(frame, conf=0.5)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        labels_idx = result.boxes.cls
        confs = result.boxes.conf
        for xyxy, label_idx, conf in zip(xyxys, labels_idx, confs):
            label = model.names[int(label_idx)]
            color = colors[int(label_idx)]  # Get the color for this class index
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
    cv2.imshow("Detecting Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
