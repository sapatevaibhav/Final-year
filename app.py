import streamlit as st
import cv2
import numpy as np
import imutils
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------
# Helper Functions (Preprocessing)
# ------------------------------

def Crop_image(img):
    img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    _, img_thresh = cv2.threshold(gray_img, 45, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(img_thresh, None, iterations=2)
    img_dilate = cv2.dilate(img_erode, None, iterations=2)
    cnts = cv2.findContours(img_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)
    img_crop = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
    return img_cnt, img_crop

def Image_PreProcessed(img, crop_func=None, target_size=(224,224)):
    img = img.copy()
    if crop_func is not None:
        _, img = crop_func(img)
    img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = (img/255.0).astype('float32')
    return img

# ------------------------------
# Load the Pretrained Model
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_my_model():
    model = load_model("Best_Model_On_Partial.h5")
    return model

model = load_my_model()

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Brain Tumor Detector", page_icon=":brain:", layout="wide")
st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload an MRI scan to detect potential brain tumors.")

# File Upload Section
with st.expander("📁 Upload MRI Scan", expanded=True):
    uploaded_file = st.file_uploader("Choose an MRI image (JPEG/PNG format)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    tab1, tab2, tab3 = st.tabs(["Original Scan", "Processing Steps", "Diagnosis Report"])
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    with tab1:
        st.subheader("Original MRI Scan")
        st.image(img_rgb, use_container_width=True, caption="Uploaded MRI Image")

    with tab2:
        st.subheader("Image Processing Pipeline")
        with st.spinner("Processing..."):
            img_cnt, img_crop = Crop_image(img_rgb)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_cnt, caption="Tumor Contour", use_container_width=True)
        with col2:
            st.image(img_crop, caption="Cropped Tumor Region", use_container_width=True)

    with tab3:
        st.subheader("Diagnosis Analysis")
        with st.spinner("Running AI diagnosis..."):
            processed_img = Image_PreProcessed(img_rgb, crop_func=Crop_image)
            input_img = np.expand_dims(processed_img, axis=0)
            predictions = model.predict(input_img)
            pred_idx = np.argmax(predictions, axis=-1)[0]
            pred_score = np.max(predictions)
            img_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
            pred_label = img_labels[pred_idx]

        st.markdown("### AI Diagnosis Result")
        if pred_label == 'notumor':
            st.success("🏆 **No Tumor Detected**")
            st.balloons()
        else:
            st.error(f"⚠️ **Potential Tumor Detected**: {pred_label.capitalize()}")
        st.markdown(f"**Confidence Level:** {pred_score*100:.2f}%")
        st.progress(int(pred_score * 100))

        # Tumor Highlighting
        with st.expander("📸 **Tumor Region Highlighted**"):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img_rgb)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(img_rgb, [contour], -1, (255, 0, 0), 2)  # Red boundary
            ax.imshow(img_rgb)
            ax.axis("off")
            st.pyplot(fig)

        with st.expander("Detailed Analysis Report"):
            st.markdown("### Prediction Breakdown")
            for label, score in zip(img_labels, predictions[0]):
                st.markdown(f"- **{label.capitalize()}**: {score*100:.2f}%")
