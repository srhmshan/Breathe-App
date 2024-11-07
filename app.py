import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

st.title("Live Object Detection with YOLOv8")

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, conf=confidence_threshold)
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame)

    cap.release()
else:
    st.write("Click the checkbox to start the webcam.")
