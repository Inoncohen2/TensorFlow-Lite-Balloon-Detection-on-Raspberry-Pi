import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import RPi.GPIO as GPIO
from collections import deque
import os
import sys
import ctypes

# Loading libjpeg library
libc = ctypes.CDLL(None)
# Suppressing errors from libjpeg
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
libc.fopen.restype = ctypes.c_void_p
devnull = libc.fopen(b'/dev/null', b'w')
ctypes.memmove(ctypes.byref(c_stderr), ctypes.byref(ctypes.c_void_p(devnull)), ctypes.sizeof(ctypes.c_void_p))

# Define pins for servo motors and laser
panServo = 23  # GPIO pin for horizontal servo
tiltServo = 18  # GPIO pin for vertical servo
laserPin = 24  # GPIO pin for laser pointer

GPIO.setmode(GPIO.BCM)
GPIO.setup(laserPin, GPIO.OUT)
GPIO.setup(panServo, GPIO.OUT)
GPIO.setup(tiltServo, GPIO.OUT)

# PWM settings for servo
pan_pwm = GPIO.PWM(panServo, 50)
tilt_pwm = GPIO.PWM(tiltServo, 50)
pan_pwm.start(0)  # Start with PWM off
tilt_pwm.start(0)  # Start with PWM off

# Global variables for motor control
current_pan = 160  # Initial angle
current_tilt = 45  # Initial angle
last_move_time = 0
MOVE_INTERVAL = 0.0005  # Minimum time between movements
PAN_DEADZONE = 0.5  # Dead zone for horizontal angle
TILT_DEADZONE = 0.5  # Dead zone for vertical angle

# System state tracking variables
last_x_pos = None
last_y_pos = None
POSITION_CHANGE_THRESHOLD = 20  # Minimum pixel change to be considered movement
ON_TARGET_THRESHOLD = 10  # Pixel threshold for "on target" status
last_target_status = False  # Previous laser on target status
last_detection_status = None  # Previous balloon detection status
PROCESS_EVERY_N_FRAMES = 2  # Process every second frame

# Variables for storing last circle position
last_circle_center = None
last_circle_radius = None

# Queue for storing object center points
x_center_queue = deque(maxlen=3)
y_center_queue = deque(maxlen=3)

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        self.stream.set(cv2.CAP_PROP_FPS, framerate)  # Set FPS
        
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

def smooth_servo_move(pwm, current_angle, target_angle):
    new_angle = current_angle + (target_angle - current_angle) * 0.97
    duty_cycle = (2.0 + (new_angle / 180.0) * 10.0)
    pwm.ChangeDutyCycle(duty_cycle)
    return new_angle

def map_value(value, min_in, max_in, min_out, max_out):
    return min_out + (max_out - min_out) * ((value - min_in) / (max_in - min_in))

def mapServoPosition(x_avg, y_avg, frame_width, frame_height):
    global current_pan, current_tilt, last_move_time
    current_time = time.time()
    
    if current_time - last_move_time < MOVE_INTERVAL:
        return

    target_pan = int(map_value(x_avg, 85, 738, 167, 129))
    target_tilt = int(map_value(y_avg, 29, 520, 33, 63))

    pan_error = abs(target_pan - current_pan)
    tilt_error = abs(target_tilt - current_tilt)
    
    angles_changed = False

    if pan_error > PAN_DEADZONE:
        current_pan = smooth_servo_move(pan_pwm, current_pan, target_pan)
        angles_changed = True
    else:
        pan_pwm.ChangeDutyCycle(0)
    
    if tilt_error > TILT_DEADZONE:
        current_tilt = smooth_servo_move(tilt_pwm, current_tilt, target_tilt)
        angles_changed = True
    else:
        tilt_pwm.ChangeDutyCycle(0)
    
    if angles_changed:
        print(f"\nServo angles: Pan={current_pan:.1f}°, Tilt={current_tilt:.1f}°")
    
    last_move_time = current_time

def control_laser(state):
    GPIO.output(laserPin, GPIO.HIGH if state else GPIO.LOW)

def check_on_target(x_center, y_center, frame_center_x, frame_center_y):
    """
    Check if laser is focused on balloon center
    """
    x_diff = abs(x_center - frame_center_x)
    y_diff = abs(y_center - frame_center_y)
    return x_diff < ON_TARGET_THRESHOLD and y_diff < ON_TARGET_THRESHOLD

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='Sample_TFLite_model')
parser.add_argument('--graph', help='Name of the .tflite file', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH', default='800x600')
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

# Initialize FPS variables
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=60).start()
time.sleep(1)

frame_count = 0
try:
    while True:
        frame_count += 1
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
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

            balloon_detected = False
            largest_area = 0
            largest_box = None
            current_time = time.time()

            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                    if labels[int(classes[i])] == 'balloon':
                        balloon_detected = True
                        ymin = int(max(1, (boxes[i][0] * imH)))
                        xmin = int(max(1, (boxes[i][1] * imW)))
                        ymax = int(min(imH, (boxes[i][2] * imH)))
                        xmax = int(min(imW, (boxes[i][3] * imW)))
                        area = (xmax - xmin) * (ymax - ymin)

                        if area > largest_area:
                            largest_area = area
                            largest_box = (xmin, ymin, xmax, ymax)

            control_laser(balloon_detected)

            if largest_box is not None:
                if last_detection_status is False:
                    print("\nBalloon detected!")
                last_detection_status = True
                
                xmin, ymin, xmax, ymax = largest_box
                x_center = (xmin + xmax) // 2
                y_center = (ymin + ymax) // 2
                radius = int(((xmax - xmin) + (ymax - ymin)) / 4)
                
                last_circle_center = (x_center, y_center)
                last_circle_radius = radius

                position_changed = False
                if last_x_pos is None or last_y_pos is None:
                    print(f"Position: X={x_center}, Y={y_center}")
                    position_changed = True
                else:
                    if (abs(x_center - last_x_pos) > POSITION_CHANGE_THRESHOLD or 
                        abs(y_center - last_y_pos) > POSITION_CHANGE_THRESHOLD):
                        print(f"\nBalloon moved to new position: X={x_center}, Y={y_center}")
                        position_changed = True

                last_x_pos = x_center
                last_y_pos = y_center

                x_center_queue.append(x_center)
                y_center_queue.append(y_center)

                x_avg = int(np.mean(x_center_queue))
                y_avg = int(np.mean(y_center_queue))

                frame_center_x = imW // 2
                frame_center_y = imH // 2
                on_target = check_on_target(x_avg, y_avg, frame_center_x, frame_center_y)
                
                if on_target != last_target_status:
                    if on_target:
                        print("\n!!! Laser centered on balloon !!!")
                        print(f"Deviation: X={abs(x_avg - frame_center_x)}, Y={abs(y_avg - frame_center_y)} pixels")
                    else:
                        print("\nLaser lost target focus")
                    last_target_status = on_target

                mapServoPosition(x_avg, y_avg, imW, imH)
            else:
                if last_detection_status is True:
                    print("\nBalloon out of view")
                    last_circle_center = None
                    last_circle_radius = None
                elif last_detection_status is None:
                    print("\nNo balloon detected in frame")
                
                last_detection_status = False
                last_x_pos = None
                last_y_pos = None
                last_target_status = False
                
                x_center_queue.clear()
                y_center_queue.clear()
                pan_pwm.ChangeDutyCycle(0)
                tilt_pwm.ChangeDutyCycle(0)

        if last_circle_center is not None and last_circle_radius is not None:
            cv2.circle(frame, last_circle_center, last_circle_radius, (0, 255, 0), 2)

        #cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('BalloonHunterPi', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    print("Cleaning up...")
    cv2.destroyAllWindows()
    videostream.stop()
    
    pan_pwm.ChangeDutyCycle(0)
    tilt_pwm.ChangeDutyCycle(0)
    time.sleep(0.1)
    pan_pwm.stop()
    tilt_pwm.stop()
    
    GPIO.cleanup()
    print("Cleanup completed")
