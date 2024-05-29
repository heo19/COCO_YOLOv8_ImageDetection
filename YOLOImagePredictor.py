from ultralytics import YOLO
from PIL import  Image
import os
import torch

class YOLOImagePredictor:
    def __init__(self, model_path, directory, conf):
        self.model = YOLO(model_path)
        self.directory = directory
        self.conf = conf
        self.save_dir = "predictionResults"
    
    def predict_and_save(self):
        results = self.model.predict(self.directory, conf=self.conf)
        
        for i, result in enumerate(results):
            for r in result:
                im_array = result.plot()
                im = Image.fromarray(im_array[..., ::-1])
                original_filename = os.listdir(self.directory)[i]
                result_filename = os.path.splitext(original_filename)[0] + f'_result_.jpg'
                im.save(os.path.join(self.save_dir, result_filename))
        
        print(f"Predictions saved in {self.save_dir}" +
              f" with confidence threshold {self.conf}" +
              f" on images from {self.directory}")

model_path = "yolov8n.pt"
directory = "testingSamples"
cocoPredictor = YOLOImagePredictor(model_path, directory, 0.5)
cocoPredictor.predict_and_save()