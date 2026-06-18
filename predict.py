import json
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

MODEL_PATH = "models/alzheimer_model.keras"
CLASS_FILE = "models/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------------------------------
# LOAD CLASS NAMES
# -------------------------------------------------------

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = json.load(f)

print("\nLoaded Classes:")
print(CLASS_NAMES)

# -------------------------------------------------------
# PREPROCESS IMAGE
# -------------------------------------------------------

IMG_SIZE = (224, 224)


def preprocess_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img = np.array(image, dtype=np.float32)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img


# -------------------------------------------------------
# PREDICT
# -------------------------------------------------------

def predict_image(image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_label = CLASS_NAMES[predicted_index]

    return predicted_label


# -------------------------------------------------------
# OPTIONAL TEST
# -------------------------------------------------------

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("\nPrediction :", result)