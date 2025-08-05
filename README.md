# Image Classifier API + Streamlit UI

This is a simple end-to-end ML project that lets you upload fashion images and get predictions via a trained CNN model. Built with **FastAPI**, **Streamlit**, and **TensorFlow**.

---

## Live URLs

- **FastAPI Backend**: https://www.youtube.com/watch?v=PqbDMhGldbc

- **FastAPI Backend**: https://fashion-prediction-backend.onrender.com
- **Streamlit UI**: https://fashion-prediction-frontend.onrender.com

---

## Project Description

This app classifies images of clothing into 10 categories using a Convolutional Neural Network trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. You can:

- Predict individual clothing items
- Retrain the model with new data (via UI or API)
- Simulate load using Locust

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourname/fashion-api.git
cd fashion-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI locally
```bash
uvicorn src.prediction:app --reload
```

### 4. Run Streamlit UI locally
```bash
streamlit run src/app.py
```

