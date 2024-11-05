
# TensorFlow Lite Object Detection on Raspberry Pi

## Real-Time Object Detection Project Using TensorFlow Lite on Raspberry Pi

This project utilizes the TensorFlow Lite model to perform object detection in real-time using a camera connected to a Raspberry Pi. The project includes a general object detection model and a specific implementation for detecting balloons, displaying the confidence percentage above each detected object.

---

## General Object Detection

### Step 1: Download the Project

Download the project code from GitHub and set up the project directory:

```bash
git clone https://github.com/Inoncohen2/TensorFlow-Lite-Balloon-Detection-on-Raspberry-Pi.git
mv TensorFlow-Lite-Balloon-Detection-on-Raspberry-Pi Balloon_Detection
cd Balloon_Detection
```

### Step 2: Set Up a Virtual Environment

Install `virtualenv` and create a virtual environment for the project:

```bash
sudo pip3 install virtualenv 
python3 -m venv balloon_env
source balloon_env/bin/activate
```

### Step 3: Install Dependencies

Install all necessary dependencies for the project, including TensorFlow Lite and OpenCV, by running the setup script:

```bash
bash get_pi_requirements.sh
```

### Step 4: Run the General Object Detection Model

Test the object detection functionality and confirm that the camera is working by running the following code:

```bash
python3 object_detection_webcam.py
```

This will start the camera feed and detect various objects based on the trained model.

---

## Balloon Detection

### Step 5: Run the Balloon Detection Model

To specifically detect balloons, ensure you have the necessary model files in the `Sample_TFLite_model` directory. You can run the following code to start the balloon detection:

```bash
python3 balloon_detector.py
```

This script will utilize a trained model specifically for balloon detection and display the detection results.

---

### System Requirements

- **Raspberry Pi** with Raspbian OS
- USB Camera or Raspberry Pi Camera

### Notes

- Ensure that the camera is connected and that the virtual environment is active before running the code.
- All rights reserved to Inon Cohen.

---

Good luck!
