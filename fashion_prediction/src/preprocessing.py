from fastapi import Request, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from pydantic import BaseModel
from typing import List
import shutil, os


UPLOAD_DIR = "data/retrain"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved.append(file.filename)
    return {"saved_files": saved}


class RetrainPayload(BaseModel):
    images: List[List[float]]  # Each image is a flattened list of 784 pixels
    labels: List[int]

@app.post("/retrain")
async def retrain_model(payload: RetrainPayload):
    # Load original FashionMNIST
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_train = x_train[..., np.newaxis]

    # Process new data
    new_images = np.array(payload.images).reshape(-1, 28, 28, 1)
    new_labels = np.array(payload.labels)

    # Combine
    x_combined = np.concatenate((x_train, new_images), axis=0)
    y_combined = np.concatenate((y_train, new_labels), axis=0)

    # Load model
    model = load_model("../models/fashion_cnn.h5")

    # Retrain
    model.fit(x_combined, y_combined, epochs=5)
    model.save("../models/fashion_cnn.h5")

    return {"message": "Model retrained with new CSV data"}


import time

start_time = time.time()

@app.get("/status")
def status():
    uptime = round(time.time() - start_time)
    return {
        "model_version": "v1",
        "uptime_secs": uptime,
        "note": "Use /predict to make predictions. /retrain to improve."
    }