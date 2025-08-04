from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import os
from preprocessing import preprocess_image
import time
import zipfile
import io
import os
import re
from PIL import Image
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


start_time = time.time()

UPLOAD_DIR = "data/retrain"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
model = load_model("../models/cifar10_cnn.h5")

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    return {
        "class": class_names[class_index],
        "confidence": float(np.max(prediction))
    }

import traceback

@app.post("/retrain")
async def retrain_from_zip(zip_file: UploadFile = File(...), epochs: int = 1):
    try:
        images = []
        labels = []

        zip_bytes = await zip_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for name in archive.namelist():
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                match = re.match(r"(\d+)", os.path.basename(name))
                if not match:
                    continue
                label = int(match.group(1))

                with archive.open(name) as file:
                    img = Image.open(file).convert("RGB").resize((32, 32))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)

        if not images:
            return {"error": "No valid images found in zip."}

        new_images = np.array(images).reshape(-1, 32, 32, 3)
        new_labels = np.array(labels).reshape(-1, 1)

        (x_train, y_train), _ = cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0

        x_combined = np.concatenate((x_train, new_images))
        y_combined = np.concatenate((y_train, new_labels))

        print(f"Retraining with shape: {x_combined.shape}, Labels: {y_combined.shape}, Epochs: {epochs}")
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_combined, y_combined, epochs=epochs)
        model.save("../models/cifar10_cnn.h5")

        return {"message": f"Retrained successfully"}

    except Exception as e:
        print("Something went wrong:")
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/metrics")
def get_model_metrics():
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0

    eval_result = model.evaluate(x_test, y_test, verbose=0)
    return {
        "loss": float(eval_result[0]),
        "accuracy": float(eval_result[1])
    }

@app.get("/status")
def status():
    uptime = round(time.time() - start_time)
    return {
        "uptime_secs": uptime,
        "note": "Use /predict to make predictions. /retrain to improve."
    }