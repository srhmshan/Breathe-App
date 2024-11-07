import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

model = YOLO('best.pt')

classes = {0: 'Blue', 1: 'Green', 2: 'Orange', 3: 'Triflow'}

st.title("Object Detection with YOLOv8 - Custom Classes")

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
        class_id = int(result.cls)
        st.write(classes.get(class_id, "Unknown"))
