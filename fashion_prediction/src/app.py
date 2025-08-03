import streamlit as st
import pandas as pd
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fashion Classifier", page_icon="🧥")
st.title("Fashion Item Classifier")

uploaded_file = st.file_uploader("Upload an image of clothing...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("Sending to model...")
        response = requests.post(
            API_URL,
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
        else:
            st.error("Prediction failed. Check API logs.")

st.title("📦 Retrain Model with New CSV Data")

csv_file = st.file_uploader("Upload a CSV with flattened image pixels and a 'label' column", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)

    if "label" not in df.columns:
        st.error("❌ CSV must contain a 'label' column.")
    else:
        st.success(f"✅ {len(df)} records loaded.")
        if st.button("Retrain Model"):
            # Convert df except label to list of lists (images)
            images = df.drop(columns=["label"]).values.tolist()
            labels = df["label"].tolist()
            payload = {"images": images, "labels": labels}

            with st.spinner("Retraining..."):
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    st.success("✅ Model retrained successfully.")
                else:
                    st.error(f"❌ Error: {res.text}")

