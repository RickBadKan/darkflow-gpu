from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.1, "gpu":0.7}

tfnet = TFNet(options)

videocv = cv2.VideoCapture(0)
result = tfnet.return_predict(videocv)
print(result)