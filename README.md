
# People Counter and Car Counter using YOLOv8n and SORT 
This Python script utilizes the YOLO object detection model (implemented with the YOLOv5 architecture) to count people in a video stream. It combines YOLOv5's object detection capabilities with the Simple Online and Realtime Tracking (SORT) algorithm for multi-object tracking. The script reads a video file or camera stream, detects people in each frame, and counts the number of people passing through predefined counting lines.

# Prerequisites
. Python 3.x
. OpenCV (cv2)
. NumPy
. Ultralytics' YOLO package (https://github.com/ultralytics/yolov8)
. cvzone package (https://github.com/cvzone/cvzone)

# Installation Of important library 
. cvzone==1.5.6
, ultralytics==8.0.26
, hydra-core>=1.2.0
, matplotlib>=3.2.2
, numpy>=1.18.5
, opencv-python==4.5.4.60
, Pillow>=7.1.2
, PyYAML>=5.3.1
, requests>=2.23.0
, scipy>=1.4.1
, torch>=1.7.0
, torchvision>=0.8.1
, tqdm>=4.64.0
, filterpy==1.4.5
, scikit-image==0.19.3
, lap==0.4.0

## Process of making the project
1) Clone this repository.
2) Install the required Python packages using pip which are listed above
3) pip install -r requirements.txt
4) Download the YOLOv8n weights file (yolov8n.pt) from the Ultralytics repository: 
5) Place the downloaded weights file in the "Yolo_Weight" directory within the project folder.

# Usage
1) Run the script by executing python main.py.
2) The script will open a window showing the video stream with real-time people counting.
3) Press any key to exit the program.

# Configuration
1) The script is configured to count only people (class "person") with a confidence threshold of 0.3. You can adjust the confidence threshold and class filtering in the code to detect and count other objects.
2) The counting lines are defined by two points for upper and lower limits in the frame. You can adjust these points in the limitsUp and limitsDown variables.

# License
This project is licensed under the MIT License.


