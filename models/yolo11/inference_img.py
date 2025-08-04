# Run this script from the parent directory
# python ./src/inference_img.py

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# load model
model_path = "./data/yolo_model.pt"
model = YOLO(model_path)

# run inference
img = cv2.imread("./src/test_img.jpg")

results = model(img)
annotator = Annotator(img)
for r in results:
    boxes = r.boxes
    for box in boxes:
        bb = box.xyxy[0]  # get box coordinates in (left, top, right, bottom)
        class_name = model.names[int(box.cls)]
        annotator.box_label(bb, class_name)
        
cv2.imwrite('./data/output.png', img)
