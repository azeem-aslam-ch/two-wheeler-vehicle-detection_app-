import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO

# Apply dark mode styling
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: white; }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .css-1cpxqw2 edgvbvh3 { color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("🛵 YOLOv8 Two-Wheeler Detector")

model = YOLO("https://huggingface.co/azeemaslam/yolov8-best/resolve/main/best.pt")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((640, 640))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting..."):
                st.image("https://media.giphy.com/media/swhRkVYLJDrCE/giphy.gif", width=150)
                img_array = np.array(image)
                results = model(img_array)
                res_plotted = results[0].plot()

            st.success("✅ Detection complete!")
            st.image(res_plotted, caption="Detection Result", use_column_width=True)

            buf = BytesIO()
            Image.fromarray(res_plotted).save(buf, format="PNG")
            st.download_button("📥 Download Result", buf.getvalue(), file_name="detection.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.subheader("📸 Capture from webcam")
camera_image = st.camera_input("Take a photo")
if camera_image:
    try:
        cam_img = Image.open(camera_image).convert("RGB").resize((640, 640))
        st.image(cam_img, caption="Captured Image", use_column_width=True)

        if st.button("📸 Detect from Webcam"):
            with st.spinner("Processing..."):
                st.image("https://media.giphy.com/media/swhRkVYLJDrCE/giphy.gif", width=150)
                cam_array = np.array(cam_img)
                results = model(cam_array)
                res_cam = results[0].plot()
            st.success("✅ Done!")
            st.image(res_cam, caption="Detection Result", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
