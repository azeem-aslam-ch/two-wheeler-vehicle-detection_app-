import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("🧠 YOLOv8 Custom Object Detection")

# Load model from a hosted URL (e.g. Google Drive direct link or Hugging Face)
model = YOLO("model = YOLO("https://huggingface.co/azeemaslam/yolov8-best/resolve/main/best.pt")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Detect Objects"):
        img_array = np.array(image)
        results = model(img_array)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Detection Result', use_column_width=True)
