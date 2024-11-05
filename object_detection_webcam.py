"""
Object Detection Project using TensorFlow Lite
Author: Inon Cohen
© All rights reserved

This code detects objects using a TensorFlow Lite model in real-time video from the camera. 
It is adapted to locate and label 80 different types of objects, and to work with a given model path.

Usage:
1. Place the .tflite model and labelmap.txt files in the "Sample_TFLite_model" directory.
2. Run the code to detect objects and display the confidence score above each detection.

Requirements:
- OpenCV
- TensorFlow Lite (or tflite_runtime)
"""

import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='Sample_TFLite_model')
parser.add_argument('--graph', help='Name of the .tflite file', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold', default=0.7)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH', default='1280x720')
parser.add_argument('--edgetpu', help='Use Edge TPU Accelerator', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow Lite
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and (GRAPH_NAME == 'detect.tflite'):
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, 
                            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

try:
    while True:
        frame1 = videostream.read()
        frame = frame1.copy()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                
                # Draw label and confidence score
                label = labels[int(classes[i])]
                confidence = f'{int(scores[i] * 100)}%'
                label_text = f'{label}: {confidence}'
                label_y = ymin - 10 if ymin - 10 > 10 else ymin + 20
                cv2.putText(frame, label_text, (xmin, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    cv2.destroyAllWindows()
    videostream.stop()
