import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from src.statistical_features import calculate_features, FEATURE_NAMES
from src.sfmov import create_original_sfmov


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "covid_baseline_mobilenetv2.keras"
)

RULE_TREE_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "covid",
    "rule_fidelity",
    "best_rule_fidelity_tree.pkl"
)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

# Exact Top-9 features from the completed ZFMIS experiment
TOP_9_FEATURES = [
    "percentile_75",
    "iqr",
    "autocorrelation",
    "rms",
    "energy",
    "snr",
    "coefficient_variation",
    "maximum",
    "range"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Explainable Deep Learning Framework")

st.markdown(
    """
    ### Transparent Medical Image Diagnosis

    This research prototype combines **MobileNetV2**, **Grad-CAM**,
    **SFMOV**, **statistical feature analysis**, **ZFMIS feature
    selection**, and a **rule-fidelity surrogate model** to provide
    both prediction and explanation.
    """
)

st.warning(
    "⚠️ Research prototype only. This application is intended for "
    "educational and research purposes and is **not a clinical "
    "diagnostic system**."
)

st.info(
    "Current deployed model: **COVID-19 Radiography Database — "
    "4-class chest X-ray classification**."
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_main_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_resource
def load_rule_model():

    if not os.path.exists(RULE_TREE_PATH):
        return None

    with open(
        RULE_TREE_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    model = load_main_model()

except Exception as error:

    st.error(
        "The MobileNetV2 model could not be loaded."
    )

    st.exception(error)

    st.stop()


rule_model = load_rule_model()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Exact MobileNetV2 preprocessing
    image_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    processed_image
):

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_index,
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# FIND GRAD-CAM LAYER
# ============================================================

@st.cache_resource
def find_gradcam_layer(model):

    # Exact layer used in completed research experiments
    try:

        return model.get_layer(
            "out_relu"
        )

    except Exception:

        pass

    # Fallback: find the last 4-D feature-map layer
    for layer in reversed(model.layers):

        try:

            shape = layer.output.shape

            if len(shape) == 4:
                return layer

        except Exception:

            continue

    raise RuntimeError(
        "Could not find a suitable Grad-CAM layer."
    )


gradcam_layer = find_gradcam_layer(
    model
)


# ============================================================
# FIND 128-D DEEP FEATURE LAYER
# ============================================================

@st.cache_resource
def find_deep_feature_layer(model):

    for layer in reversed(model.layers):

        try:

            shape = layer.output.shape

            if (
                len(shape) == 2
                and shape[-1] == 128
            ):

                return layer

        except Exception:

            continue

    raise RuntimeError(
        "Could not find the 128-dimensional "
        "deep feature layer."
    )


deep_feature_layer = find_deep_feature_layer(
    model
)


# ============================================================
# FEATURE MODELS
# ============================================================

@st.cache_resource
def create_deep_feature_model(
    model,
    layer
):

    return tf.keras.Model(
        inputs=model.input,
        outputs=layer.output
    )


@st.cache_resource
def create_gradcam_model(
    model,
    layer
):

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            layer.output,
            model.output
        ]
    )


deep_feature_model = create_deep_feature_model(
    model,
    deep_feature_layer
)

gradcam_model = create_gradcam_model(
    model,
    gradcam_layer
)


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam_map(
    processed_image,
    class_index
):

    with tf.GradientTape() as tape:

        feature_maps, predictions = (
            gradcam_model(
                processed_image,
                training=False
            )
        )

        target_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        target_score,
        feature_maps
    )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        feature_maps
        * weights[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ],
        axis=-1
    )

    cam = tf.nn.relu(
        cam
    )

    cam = cam[0].numpy()

    cam_min = np.min(cam)
    cam_max = np.max(cam)

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min + 1e-8
    )

    cam_tensor = tf.convert_to_tensor(
        cam[..., np.newaxis],
        dtype=tf.float32
    )

    cam_tensor = tf.image.resize(
        cam_tensor,
        (224, 224)
    )

    cam = tf.squeeze(
        cam_tensor
    ).numpy()

    return cam


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_map(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    minimum = np.min(x)
    maximum = np.max(x)

    return (
        x - minimum
    ) / (
        maximum - minimum + 1e-8
    )


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def create_overlay(
    image,
    explanation_map
):

    import matplotlib.pyplot as plt

    explanation_map = normalize_map(
        explanation_map
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        image
    )

    ax.imshow(
        explanation_map,
        cmap="jet",
        alpha=0.45
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )

    return fig


# ============================================================
# EXPLANATION COMPARISON
# ============================================================

def calculate_pearson(
    map1,
    map2
):

    map1 = normalize_map(
        map1
    ).flatten()

    map2 = normalize_map(
        map2
    ).flatten()

    correlation, _ = pearsonr(
        map1,
        map2
    )

    return float(
        correlation
    )


def calculate_ssim(
    map1,
    map2
):

    map1 = normalize_map(
        map1
    )

    map2 = normalize_map(
        map2
    )

    return float(
        ssim(
            map1,
            map2,
            data_range=1.0
        )
    )


def calculate_top20_iou(
    map1,
    map2
):

    map1 = normalize_map(
        map1
    )

    map2 = normalize_map(
        map2
    )

    threshold1 = np.percentile(
        map1,
        80
    )

    threshold2 = np.percentile(
        map2,
        80
    )

    mask1 = (
        map1 >= threshold1
    )

    mask2 = (
        map2 >= threshold2
    )

    intersection = np.logical_and(
        mask1,
        mask2
    ).sum()

    union = np.logical_or(
        mask1,
        mask2
    ).sum()

    return float(
        intersection /
        (union + 1e-8)
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        "The application uses a trained MobileNetV2 "
        "model for four-class chest X-ray classification."
    )

    st.divider()

    st.subheader("Classes")

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name}"
        )

    st.divider()

    st.subheader("Explainability")

    st.write(
        "• Grad-CAM"
    )

    st.write(
        "• Original SFMOV"
    )

    st.write(
        "• Statistical Features"
    )

    st.write(
        "• ZFMIS"
    )

    st.write(
        "• Rule-Fidelity Model"
    )

    st.divider()

    st.caption(
        "Research prototype — not for clinical diagnosis."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    help=(
        "Upload a chest X-ray image for research "
        "demonstration."
    )
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        ## How to use

        1. Upload a chest X-ray image.
        2. Run the trained deep-learning model.
        3. View the predicted class and confidence.
        4. Inspect Grad-CAM and SFMOV explanations.
        5. Review statistical features and ZFMIS.
        6. Compare the rule-fidelity prediction.

        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as error:

    st.error(
        "The uploaded file could not be read."
    )

    st.exception(error)

    st.stop()


# ============================================================
# INPUT IMAGE
# ============================================================

st.subheader(
    "1. Input Medical Image"
)

col1, col2 = st.columns(
    [1, 2]
)

with col1:

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

with col2:

    st.markdown(
        """
        **Image successfully loaded.**

        The uploaded image is converted to RGB,
        resized to **224 × 224**, and passed through
        the MobileNetV2 preprocessing pipeline.
        """
    )


# ============================================================
# PREPROCESS
# ============================================================

processed_image = preprocess_input_image(
    image
)


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

(
    predicted_index,
    predicted_class,
    confidence,
    probabilities
) = predict_image(
    model,
    processed_image
)


# ============================================================
# PREDICTION DISPLAY
# ============================================================

st.subheader(
    "2. Deep Learning Prediction"
)

col1, col2 = st.columns(
    2
)

with col1:

    st.metric(
        "Predicted Class",
        predicted_class
    )

with col2:

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )


probability_df = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Probability (%)":
            probabilities * 100
    }
)

st.dataframe(
    probability_df.style.format(
        {
            "Probability (%)":
                "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROBABILITY BAR CHART
# ============================================================

st.bar_chart(
    probability_df.set_index(
        "Class"
    )
)


# ============================================================
# GRAD-CAM
# ============================================================

st.subheader(
    "3. Grad-CAM Explanation"
)

gradcam_map = generate_gradcam_map(
    processed_image,
    predicted_index
)

gradcam_overlay = create_overlay(
    image,
    gradcam_map
)

col1, col2, col3 = st.columns(
    3
)

with col1:

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

with col2:

    st.image(
        gradcam_map,
        caption="Grad-CAM Heatmap",
        use_container_width=True
    )

with col3:

    st.pyplot(
        gradcam_overlay,
        clear_figure=True
    )


# ============================================================
# ORIGINAL SFMOV
# ============================================================

st.subheader(
    "4. Original SFMOV Explanation"
)

feature_map_output, _ = gradcam_model(
    processed_image,
    training=False
)

feature_maps = (
    feature_map_output.numpy()
)

original_sfmov = create_original_sfmov(
    feature_maps
)

original_sfmov = normalize_map(
    original_sfmov
)

sfmov_tensor = tf.convert_to_tensor(
    original_sfmov[..., np.newaxis],
    dtype=tf.float32
)

sfmov_tensor = tf.image.resize(
    sfmov_tensor,
    (224, 224)
)

original_sfmov = tf.squeeze(
    sfmov_tensor
).numpy()

sfmov_overlay = create_overlay(
    image,
    original_sfmov
)

col1, col2, col3 = st.columns(
    3
)

with col1:

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

with col2:

    st.image(
        original_sfmov,
        caption="Original SFMOV Heatmap",
        use_container_width=True
    )

with col3:

    st.pyplot(
        sfmov_overlay,
        clear_figure=True
    )

st.caption(
    "Original SFMOV combines normalized mean, "
    "skewness and entropy maps using equal weighting."
)


# ============================================================
# GRAD-CAM VS SFMOV
# ============================================================

st.subheader(
    "5. Grad-CAM vs SFMOV Comparison"
)

pearson_value = calculate_pearson(
    gradcam_map,
    original_sfmov
)

ssim_value = calculate_ssim(
    gradcam_map,
    original_sfmov
)

iou_value = calculate_top20_iou(
    gradcam_map,
    original_sfmov
)

col1, col2, col3 = st.columns(
    3
)

with col1:

    st.metric(
        "Pearson Correlation",
        f"{pearson_value:.4f}"
    )

with col2:

    st.metric(
        "SSIM",
        f"{ssim_value:.4f}"
    )

with col3:

    st.metric(
        "Top-20% Heatmap IoU",
        f"{iou_value:.4f}"
    )

st.caption(
    "These metrics measure similarity between explanation "
    "maps. Similarity alone does not establish explanation quality."
)


# ============================================================
# 128-D DEEP FEATURES
# ============================================================

st.subheader(
    "6. Deep Feature Representation"
)

deep_features = (
    deep_feature_model.predict(
        processed_image,
        verbose=0
    )
)

deep_features = np.asarray(
    deep_features,
    dtype=np.float32
).reshape(-1)

st.write(
    f"Extracted deep features: "
    f"**{len(deep_features)}**"
)


# ============================================================
# 26 STATISTICAL FEATURES
# ============================================================

st.subheader(
    "7. Statistical Feature Extraction"
)

statistical_values = calculate_features(
    deep_features
)

statistical_values = np.asarray(
    statistical_values,
    dtype=np.float32
).reshape(-1)

if len(statistical_values) != 26:

    st.error(
        f"Expected 26 statistical features, "
        f"but received {len(statistical_values)}."
    )

    st.stop()


statistical_dict = dict(
    zip(
        FEATURE_NAMES,
        statistical_values
    )
)

statistical_df = pd.DataFrame(
    {
        "Feature": FEATURE_NAMES,
        "Value": statistical_values
    }
)

st.write(
    "Extracted statistical features: "
    f"**{len(statistical_values)}**"
)

st.dataframe(
    statistical_df.style.format(
        {
            "Value": "{:.6f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ZFMIS
# ============================================================

st.subheader(
    "8. ZFMIS Feature Selection"
)

selected_values = np.array(
    [
        statistical_dict[
            feature_name
        ]
        for feature_name in TOP_9_FEATURES
    ],
    dtype=np.float32
).reshape(
    1,
    -1
)

zmfis_df = pd.DataFrame(
    {
        "Selected Feature":
            TOP_9_FEATURES,

        "Value":
            selected_values.flatten()
    }
)

st.write(
    "ZFMIS selected **9 features** "
    "from the original 26."
)

st.dataframe(
    zmfis_df.style.format(
        {
            "Value": "{:.6f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RULE-FIDELITY MODEL
# ============================================================

st.subheader(
    "9. Rule-Fidelity Explanation"
)

if rule_model is None:

    st.warning(
        "The saved rule-fidelity model was not found. "
        "The deep-learning prediction remains available."
    )

else:

    try:

        rule_prediction = rule_model.predict(
            selected_values
        )

        rule_prediction = int(
            np.asarray(
                rule_prediction
            ).reshape(-1)[0]
        )

        if (
            0 <= rule_prediction
            < len(CLASS_NAMES)
        ):

            rule_class = CLASS_NAMES[
                rule_prediction
            ]

        else:

            rule_class = str(
                rule_prediction
            )


        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Deep Learning Prediction",
                predicted_class
            )

        with col2:

            st.metric(
                "Rule-Fidelity Prediction",
                rule_class
            )


        if rule_class == predicted_class:

            st.success(
                "✓ The rule-fidelity model agrees "
                "with the Deep Learning prediction."
            )

        else:

            st.warning(
                "The rule-fidelity model differs "
                "from the Deep Learning prediction."
            )


        st.info(
            "The rule-fidelity model was trained to approximate "
            "MobileNetV2 predictions. It is therefore a surrogate "
            "explanation of the deep model, not an independent "
            "clinical diagnosis."
        )

    except Exception as error:

        st.error(
            "The saved rule-fidelity model could not "
            "process the selected features."
        )

        st.exception(error)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

st.subheader(
    "10. Complete Explanation Pipeline"
)

pipeline_df = pd.DataFrame(
    {
        "Stage": [
            "Input",
            "Preprocessing",
            "Deep Learning",
            "Prediction",
            "Deep Features",
            "Statistical Features",
            "ZFMIS",
            "Rule-Fidelity",
            "Visual Explanation"
        ],

        "Result": [
            "Medical Image",
            "224 × 224 + MobileNetV2 preprocessing",
            "MobileNetV2",
            predicted_class,
            "128 features",
            "26 features",
            "9 selected features",
            "Surrogate rule model",
            "Grad-CAM + Original SFMOV"
        ]
    }
)

st.dataframe(
    pipeline_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ARCHITECTURE
# ============================================================

st.subheader(
    "Application Architecture"
)

st.code(
    """
Medical Image
      ↓
Preprocessing
      ↓
MobileNetV2
      ↓
Prediction + Confidence
      ↓
128-D Deep Features
      ↓
26 Statistical Features
      ↓
ZFMIS
      ↓
9 Selected Features
      ↓
Rule-Fidelity Model
      ↓
Surrogate Rule Explanation

Parallel Explainability:
      ↓
Grad-CAM + Original SFMOV
      ↓
Explanation Comparison
""",
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable Deep Learning Framework for Transparent "
    "Medical Image Diagnosis | Research Prototype"
)

st.caption(
    "This application does not provide medical advice or "
    "clinical diagnosis."
)