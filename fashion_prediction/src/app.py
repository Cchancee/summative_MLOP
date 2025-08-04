import streamlit as st
import requests
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

PREDICT_URL = os.getenv("PREDICT_URL")
RETRAIN_URL = os.getenv("RETRAIN_URL")
METRICS_URL = os.getenv("METRICS_URL")

st.set_page_config(page_title="Image Classifier", page_icon="🤖")
st.title("CIFAR-10 Image Classifier")

# --- Prediction ---
uploaded_file = st.file_uploader("Upload an image (32x32 RGB)...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            response = requests.post(
                PREDICT_URL,
                files={"file": uploaded_file.getvalue()}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['class']}**")
                st.info(f"Confidence: {result['confidence']:.2%}")
            else:
                st.error("Prediction failed. Check API logs.")

# --- Retrain ---
st.markdown("---")
st.title("Retrain Model with New Images")

zip_file = st.file_uploader("Upload a ZIP of labeled images (e.g., `0_image1.jpg`, `1_image2.png`)", type=["zip"])
epochs = st.slider("Epochs for Retraining", min_value=1, max_value=15, value=5)

if zip_file and st.button("Retrain Model"):
    with st.spinner("Retraining..."):
        response = requests.post(
            RETRAIN_URL,
            files={"zip_file": zip_file},
            data={"epochs": str(epochs)}
        )
        if response.status_code == 200:
            st.success(f"Model retrained: {response.json()['message']}")
        else:
            st.error(f"Retraining failed: {response.text}")


# --- Metrics ---
st.subheader("Current Model Performance")

try:
    metrics = requests.get(METRICS_URL).json()
    st.metric(label="Accuracy", value=f"{metrics['accuracy']:.2%}")
    st.metric(label="Loss", value=f"{metrics['loss']:.4f}")
except Exception as e:
    st.warning("Couldn't fetch model metrics.")
