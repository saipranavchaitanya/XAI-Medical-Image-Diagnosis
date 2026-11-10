
import os
import numpy as np
import tensorflow as tf

from .preprocessing import load_image


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# --------------------------------------------------
# DATASET CONFIGURATION
# Class order must match the original model training.
# --------------------------------------------------

DATASET_CONFIG = {
    "covid19": {
        "name": "COVID-19 Radiography Database",
        "model": "covid_baseline_mobilenetv2.keras",
        "classes": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    },

    "breast_ultrasound": {
        "name": "Breast Ultrasound",
        "model": "breast_ultrasound_mobilenetv2_best.keras",
        "classes": [
            "benign",
            "malignant"
        ]
    },

    "bttypes": {
        "name": "BTTypes",
        "model": "bttypes_mobilenetv2_best.keras",
        "classes": [
            "Benign",
            "Malignant"
        ]
    },

    "lung_colon": {
        "name": "Lung and Colon Cancer",
        "model": "lung_colon_mobilenetv2_best.keras",
        "classes": [
            "colon_aca",
            "colon_n",
            "lung_aca",
            "lung_n",
            "lung_scc"
        ]
    },

    "acrima": {
        "name": "ACRIMA",
        "model": "acrima_mobilenetv2_best.keras",
        "classes": [
            "Glaucoma",
            "Normal"
        ]
    }
}


# --------------------------------------------------
# MODEL CACHE
# --------------------------------------------------

_model_cache = {}


# --------------------------------------------------
# GET AVAILABLE DATASETS
# --------------------------------------------------

def get_available_datasets():
    return {
        key: value["name"]
        for key, value in DATASET_CONFIG.items()
    }


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def get_model(dataset_name):

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    if dataset_name not in _model_cache:

        model_filename = DATASET_CONFIG[dataset_name]["model"]

        model_path = os.path.join(
            MODEL_DIR,
            model_filename
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Check the model filename in DATASET_CONFIG."
            )

        print(f"Loading model: {model_filename}")

        model = tf.keras.models.load_model(model_path)

        _model_cache[dataset_name] = model

    return _model_cache[dataset_name]


# --------------------------------------------------
# PREDICT IMAGE
# --------------------------------------------------

def predict_image(image_path, dataset_name="covid19"):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    config = DATASET_CONFIG[dataset_name]
    class_names = config["classes"]

    model = get_model(dataset_name)

    # Preprocess image
    image = load_image(image_path)

    # Model prediction
    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model output has {len(probabilities)} classes, "
            f"but configuration contains {len(class_names)} classes."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid probabilities.")

    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]

    confidence = float(probabilities[predicted_index])

    probability_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "dataset": dataset_name,
        "dataset_name": config["name"],
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probability_dict
    }
