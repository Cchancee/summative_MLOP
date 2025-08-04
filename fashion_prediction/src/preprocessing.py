from PIL import Image
import numpy as np
from io import BytesIO

def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB").resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 32, 32, 3)
    return image_array
