
import os
import numpy as np
import tensorflow as tf

from .preprocessing import load_image


CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "covid_baseline_mobilenetv2.keras"
)


_model = None


def get_model():
    """Load the trained MobileNetV2 model once."""

    global _model

    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH
        )

    return _model


def predict_image(image_path):
    """Return predicted class, confidence and probabilities."""

    model = get_model()

    image = load_image(
        image_path
    )

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probability_dict
    }
