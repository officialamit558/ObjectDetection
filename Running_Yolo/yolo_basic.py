from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_Weight/yolov8l.pt')
results = model('Running_Yolo/images/2.jpeg' , show=True)
cv2.waitKey(0)
