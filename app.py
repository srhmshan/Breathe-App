import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

st.title("Spirometer Object Detection")

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

img_file_buffer = st.camera_input("Capture Image")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    results = model(img_array, conf=confidence_threshold)

    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects")
    
    st.write("Detected classes:")
    for result in results[0].boxes:
        st.write(model.names[int(result.cls)])
