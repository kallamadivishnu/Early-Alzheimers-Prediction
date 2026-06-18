import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("models/alzheimer_model.keras")

# Class names
CLASSES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# Colors for Streamlit
COLORS = {
    "Non Demented": "green",
    "Very Mild Demented": "orange",
    "Mild Demented": "red",
    "Moderate Demented": "darkred"
}


def preprocess_image(image):

    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)[0]

    index = np.argmax(prediction)

    label = CLASSES[index]

    confidence = float(prediction[index] * 100)

    probabilities = {}

    for i in range(len(CLASSES)):
        probabilities[CLASSES[i]] = round(float(prediction[i] * 100), 2)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "color": COLORS[label]
    }


if __name__ == "__main__":

    path = input("Enter MRI image path: ")

    result = predict_image(path)

    print("\nPrediction")
    print("----------------------------")
    print("Stage      :", result["label"])
    print("Confidence :", result["confidence"], "%")

    print("\nProbabilities")

    for k, v in result["probabilities"].items():
        print(f"{k:25} {v}%")