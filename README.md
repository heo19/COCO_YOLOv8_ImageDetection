# COCO_YOLOv8_ImageDetection

testingSamples: contains 602 images from test2017 from COCO dataset.

predictionResults: contains 602 images after detection with bbox

results.json: contains json data from predictionResults. {name, class, boxes with coordinates}

YOLOImagePredictor.py : detects object with pretrained YOLOv8

YOLOImageWebcamDetector.py : detects object real time in webcam with pretrained YOLOv8