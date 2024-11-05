
# TensorFlow Lite Balloon Detection on Raspberry Pi

## Real-Time Balloon Detection Project Using TensorFlow Lite on Raspberry Pi

This project utilizes the SSD Lite MobileNet model to detect balloons in real-time, displaying the confidence percentage above each detected balloon. Follow these instructions to set up and run the project.

---

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

### Step 4: Run the Model for Initial Testing

Test the balloon detection and confirm that the camera is working by running the following code:

```bash
python3 balloon_detector.py
```

---

### System Requirements

- **Raspberry Pi** with Raspbian OS
- USB Camera or Raspberry Pi Camera

### Notes

- Ensure that the camera is connected and that the virtual environment is active before running the code.
- All rights reserved to Inon Cohen.

---

Good luck!
