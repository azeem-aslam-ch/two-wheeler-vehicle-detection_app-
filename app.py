import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Bike and Motor Bike Detection", layout="wide")
st.title("🛵 Two-Wheeler Vehicle Detection using YOLOv8")

# Load model from Hugging Face (or local if available)
model = YOLO("https://huggingface.co/azeemaslam/yolov8-best/resolve/main/best.pt")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")  # Ensure correct format
        image = image.resize((640, 640))  # Resize for faster detection
        st.image(image, caption='Uploaded Image (Resized)', use_column_width=True)

        if st.button("🔍 Detect Objects"):
            with st.spinner("Running detection... please wait ⏳"):
                img_array = np.array(image)
                results = model(img_array)
                res_plotted = results[0].plot()
            st.success("✅ Detection complete!")
            st.image(res_plotted, caption='Detection Result', use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")


