import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Title
st.title('Orange Quality Detection (YOLO)')

# Load model
@st.cache_resource
def load_model():
    model = YOLO('D:\\Orange Fruit Quality Assesment\\yolo\\runs\\detect\\60 epoch 640x640\\weights\\best.pt')
    return model

model = load_model()

# Get class names
try:
    class_names = model.names
except AttributeError:
    class_names = [str(i) for i in range(model.model.model[-1].nc)]

# Video-based detection
st.write('## Real-Time Webcam Detection')
run = st.button('Start/Stop Video')
FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error('Could not open webcam.')
    else:
        stop = False
        while run and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error('Failed to grab frame')
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # YOLO detection
            results = model(frame_rgb)
            annotated = results[0].plot()
            # Display
            FRAME_WINDOW.image(annotated, channels='RGB', use_container_width=True)
            # Add a small delay to allow Streamlit to update
            time.sleep(0.03)
            # Check if the button is pressed again to stop
            run = st.session_state.get('run', True)
        cap.release()
else:
    st.write('Click the button to start/stop the webcam.') 