import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO, RTDETR

model_paths = {'yolov5n': './Models/YOLOV5N/weights/last.pt', 
               'yolov8n': './Models/YOLOV8N/weights/last.pt', 
               'yolov8no': './Models/YOLOV8NO/best.pt',
               'rtdetr': './Models/RTDETR/weights/last.pt'}
model_support = ['yolov5n', 'yolov8n', 'yolov8no', 'rtdetr']
class ObjectDetection:
    def __init__(self, model="yolov8n", device = None):
        if model not in model_support:
            raise Exception("Model not support")
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Dectector model is loading...")
        self.model_path = os.path.abspath(model_paths[model])
        print("Model path: ", self.model_path)
        self.model = model
        if model == "rtdetr":
            self.model = RTDETR(model_paths[model])
        else:
            self.model = YOLO(model_paths[model])
        self.device = device
        print("You are using: ", model)
        print("Dectector info: ", self.model.info())
        print("Dectector model is loaded!")
    def predict(self, frame, specific_class = None, conf = 0.3, iou = 0.5, threshold = 30, plot = False):
        """_summary_
        predict the frame and return the result
        Args:
            frame (_type_): frame to predict numpy array
            specific_class (list optional): List of class you want to extract. Defaults to None. Mean that you want to extract all class.

        Returns:
            list: list of result object with class, confidence, and bounding box
        """
        results = self.model.predict(frame, conf=conf, iou=iou, device=self.device, verbose=False)
        b_results = results[0].numpy().boxes
        clses = b_results.cls.astype(int)
        xxyys = b_results.xyxy.astype(int)
        confs = b_results.conf.astype(float)
        if specific_class is not None:
            clses = clses[np.isin(clses, specific_class)]
            xxyys = np.array([xxyys[i] for i in range(len(clses)) if (clses[i] in specific_class) and (xxyys[i][2] - xxyys[i][0] > threshold)])
        if plot:
            return xxyys, clses, results[0].plot()
        else:
            return xxyys, clses, confs

