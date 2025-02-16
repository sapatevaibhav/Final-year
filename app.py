import streamlit as st
import cv2
import numpy as np
import imutils
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# ------------------------------
# Helper Functions (Preprocessing)
# ------------------------------

def Crop_image(img):
    """
    Finds the extreme points on the image and crops the rectangular area out of them.
    Returns:
        img_cnt: the image with the drawn contour,
        img_pnt: the image with extreme points highlighted,
        img_crop: the cropped image.
    """
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
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0,0,255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0,255,0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255,0,0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255,0,0), -1)
    Add_pixel = 0
    img_crop = img[extTop[1]-Add_pixel:extBot[1]+Add_pixel, extLeft[0]-Add_pixel:extRight[0]+Add_pixel].copy()
    return img_cnt, img_pnt, img_crop

def Image_PreProcessed(img, crop_func=None, target_size=(224,224)):
    """
    Performs full image preprocessing including optional cropping, resizing, denoising,
    applying a colormap, and normalization.
    """
    img = img.copy()
    if crop_func is not None:
        _, _, img = crop_func(img)
    img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = (img/255.0).astype('float32')
    return img

# ------------------------------
# Define Labels (must match your training labels)
# ------------------------------
img_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .st-emotion-cache-1v0mbdj {border-radius: 10px;}
    .stProgress > div > div > div > div {background-color: #4CAF50;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.title("🧠 Brain Tumor Detection System")
st.markdown("""
    Upload an MRI scan to detect potential brain tumors.
    The system will analyze the image and provide diagnostic predictions.
    """)

# File Upload Section
with st.expander("📁 Upload MRI Scan", expanded=True):
    uploaded_file = st.file_uploader(
        "Choose an MRI image (JPEG/PNG format)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    # Image Processing Section
    tab1, tab2, tab3 = st.tabs(["Original Scan", "Processing Steps", "Diagnosis Report"])

    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    with tab1:
        st.subheader("Original MRI Scan")
        st.image(img_rgb, use_container_width=True, caption="Uploaded MRI Image")

    with tab2:
        try:
            st.subheader("Image Processing Pipeline")

            with st.spinner("Analyzing image structure..."):
                img_cnt, img_pnt, img_crop = Crop_image(img_rgb)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_cnt, caption="Tumor Contour Detection", use_container_width=True)
            with col2:
                st.image(img_pnt, caption="Extreme Points Identification", use_container_width=True)
            with col3:
                st.image(img_crop, caption="Cropped Region of Interest", use_container_width=True)

            with st.expander("Advanced Preprocessing Details"):
                processed_img = Image_PreProcessed(img_rgb, crop_func=Crop_image)
                disp_img = np.clip(processed_img * 255, 0, 255).astype(np.uint8)
                disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
                st.image(disp_img, caption="Final Preprocessed Image", use_container_width=True)
                st.write("Preprocessing steps include: Cropping, Resizing, Denoising, and Color Mapping")

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.warning("Please ensure the image is a clear MRI scan with proper brain structure visible.")

    with tab3:
        try:
            st.subheader("Diagnosis Analysis")

            with st.spinner("Running AI diagnosis..."):
                processed_img = Image_PreProcessed(img_rgb, crop_func=Crop_image)
                input_img = np.expand_dims(processed_img, axis=0)
                predictions = model.predict(input_img)
                pred_idx = np.argmax(predictions, axis=-1)[0]
                pred_score = np.max(predictions)
                pred_label = img_labels[pred_idx]

            st.markdown("### AI Diagnosis Result")

            if pred_label == 'notumor':
                st.success("🏆 **No Tumor Detected**")
                st.balloons()
            else:
                st.error(f"⚠️ **Potential Tumor Detected**: {pred_label.capitalize()}")

            # Confidence Meter
            st.markdown(f"**Confidence Level:** {pred_score*100:.2f}%")
            st.progress(int(pred_score * 100))

            # Detailed Report
            with st.expander("Detailed Analysis Report"):
                st.markdown("### Prediction Breakdown")
                for label, score in zip(img_labels, predictions[0]):
                    st.markdown(f"- **{label.capitalize()}**: {score*100:.2f}%")

                st.markdown("""
                    ### What This Means
                    - **Glioma**: Tumor originating in the brain's glial cells
                    - **Meningioma**: Tumor arising from the meninges
                    - **Pituitary**: Tumor in the pituitary gland
                    - **No Tumor**: No abnormal growth detected
                    """)

            st.markdown("""
                > **Note**: This AI analysis should be used as a secondary diagnostic tool.
                Always consult with a medical professional for final diagnosis.
                """)

        except Exception as e:
            st.error(f"Diagnosis error: {str(e)}")

else:
    st.info("ℹ️ Please upload an MRI scan using the uploader above to begin analysis.")
    st.markdown("### Example MRI Scans")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://example.com/glioma_sample.jpg", caption="Glioma Tumor Example", use_container_width=True)
    with col2:
        st.image("https://example.com/meningioma_sample.jpg", caption="Meningioma Tumor Example", use_container_width=True)
    with col3:
        st.image("https://example.com/no_tumor_sample.jpg", caption="Healthy Brain Example", use_container_width=True)
