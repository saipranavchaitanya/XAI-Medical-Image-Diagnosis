import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from src.prediction import DATASET_CONFIG, get_available_datasets, get_model
from src.statistical_features import calculate_features, FEATURE_NAMES
from src.sfmov import create_original_sfmov


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEATURE_ALIASES = {
    "p10": "percentile_10",
    "p25": "percentile_25",
    "p50": "percentile_50",
    "p75": "percentile_75",
}

ZFMIS_FEATURES = {
    "covid19": [
        "percentile_75", "iqr", "autocorrelation", "rms", "energy",
        "snr", "coefficient_variation", "maximum", "range",
    ],

    "breast_ultrasound": [
        "geometric_mean", "entropy", "shannon_entropy",
        "autocorrelation", "mad", "median", "percentile_50", "snr",
        "coefficient_variation", "percentile_75", "iqr", "sem", "std",
        "variance", "contrast", "maximum", "range", "mean",
    ],

    "bttypes": [
        "percentile_75", "iqr", "mean", "shannon_entropy", "entropy",
        "coefficient_variation", "snr", "geometric_mean",
        "harmonic_mean", "median", "mad", "percentile_50", "energy",
        "rms", "skewness", "autocorrelation", "kurtosis", "contrast",
    ],

    "lung_colon": [
        "entropy", "shannon_entropy", "coefficient_variation", "snr",
        "geometric_mean", "skewness", "harmonic_mean", "kurtosis",
        "iqr", "percentile_75", "autocorrelation", "mean",
    ],

    "acrima": [
        "geometric_mean", "coefficient_variation", "iqr",
    ],
}

COVID_RULE_TREE_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "covid",
    "rule_fidelity",
    "best_rule_fidelity_tree.pkl",
)

SAVED_RULE_FIDELITY = {
    "covid19": 0.5109,
    "breast_ultrasound": 0.7438017,
    "bttypes": 0.788889,
    "lung_colon": 0.871467,
    "acrima": 0.944444,
}

DATASET_DESCRIPTIONS = {
    "covid19": "Chest X-ray classification",
    "breast_ultrasound": "Breast ultrasound image classification",
    "bttypes": "Breast tissue image classification",
    "lung_colon": "Lung and colon histopathology classification",
    "acrima": "Retinal fundus image classification",
}

DATASET_MODALITIES = {
    "covid19": "Chest X-ray",
    "breast_ultrasound": "Ultrasound",
    "bttypes": "Breast tissue image",
    "lung_colon": "Histopathology",
    "acrima": "Retinal fundus image",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATASET SELECTOR
# ============================================================

available_datasets = get_available_datasets()

with st.sidebar:

    st.title("Dataset Selection")

    selected_dataset_label = st.selectbox(
        "Choose a medical image dataset",
        options=list(available_datasets.values()),
        index=0,
    )

    selected_dataset = next(
        key
        for key, value in available_datasets.items()
        if value == selected_dataset_label
    )

    class_names = DATASET_CONFIG[selected_dataset]["classes"]

    st.divider()

    st.header("About the Selected Model")

    st.write(DATASET_DESCRIPTIONS[selected_dataset])

    st.write(
        f"**Modality:** {DATASET_MODALITIES[selected_dataset]}"
    )

    st.subheader("Classes")

    for class_name in class_names:
        st.write(f"• {class_name}")

    st.divider()

    st.subheader("Explainability")

    st.write("• Grad-CAM")
    st.write("• Original SFMOV")
    st.write("• 128-D deep feature representation")
    st.write("• 26 statistical features")
    st.write("• Dataset-specific ZFMIS")
    st.write("• Rule-Fidelity experiment information")

    st.divider()

    st.caption(
        "Research prototype — not for clinical diagnosis."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Explainable Deep Learning Framework")

st.markdown("### Transparent Medical Image Diagnosis")

st.markdown(
    f"""
    **Selected dataset:** {selected_dataset_label}

    **Task:** {DATASET_DESCRIPTIONS[selected_dataset]}
    """
)

st.warning(
    "⚠️ Research prototype only. This application is intended for "
    "educational and research purposes. It is not clinically validated "
    "and must not be used as a clinical diagnostic system."
)

if selected_dataset == "lung_colon":

    st.warning(
        "The Lung & Colon class-index order currently follows "
        "src/prediction.py. Its original training class-index mapping "
        "has not yet been independently verified against training metadata."
    )


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model = get_model(selected_dataset)

except Exception as error:

    st.error(
        "The selected MobileNetV2 model could not be loaded."
    )

    st.exception(error)

    st.stop()


class_count = len(class_names)

model_output_count = int(
    model.output_shape[-1]
)

if model_output_count != class_count:

    st.error(
        f"Model output has {model_output_count} classes, "
        f"but the configuration contains {class_count} class names."
    )

    st.stop()


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

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    selected_model,
    processed_image,
    selected_classes
):

    predictions = selected_model.predict(
        processed_image,
        verbose=0
    )

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float64
    )

    if len(probabilities) != len(selected_classes):

        raise ValueError(
            f"Model returned {len(probabilities)} probabilities, "
            f"but {len(selected_classes)} class names are configured."
        )

    if not np.all(np.isfinite(probabilities)):

        raise ValueError(
            "The model returned invalid probability values."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = selected_classes[
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
# DEEP FEATURE LAYER
# ============================================================

def find_deep_feature_layer(selected_model):

    for layer in reversed(selected_model.layers):

        try:

            shape = tuple(
                layer.output.shape
            )

            if (
                len(shape) == 2
                and shape[-1] == 128
            ):

                return layer

        except Exception:

            continue

    raise RuntimeError(
        "Could not find the 128-dimensional deep feature layer."
    )


@st.cache_resource(show_spinner=False)
def create_deep_feature_model(_selected_model):

    feature_layer = find_deep_feature_layer(
        _selected_model
    )

    return tf.keras.Model(
        inputs=_selected_model.inputs,
        outputs=feature_layer.output
    )


# ============================================================
# GRAD-CAM MODEL CONSTRUCTION
# ============================================================

@st.cache_resource(show_spinner=False)
def create_gradcam_model(
    _selected_model,
    dataset_key
):

    try:

        feature_layer = _selected_model.get_layer(
            "out_relu"
        )

        return tf.keras.Model(
            inputs=_selected_model.inputs,
            outputs=[
                feature_layer.output,
                _selected_model.output
            ]
        )

    except Exception:

        pass

    backbone = None

    for layer in _selected_model.layers:

        if layer.name == "mobilenetv2_1.00_224":

            backbone = layer

            break

    if backbone is None:

        raise RuntimeError(
            f"Could not locate the Grad-CAM backbone for {dataset_key}."
        )

    model_input_shape = _selected_model.input_shape[1:]

    gradcam_input = tf.keras.Input(
        shape=model_input_shape
    )

    feature_maps = backbone(
        gradcam_input,
        training=False
    )

    classifier_output = feature_maps

    backbone_position = _selected_model.layers.index(
        backbone
    )

    for layer in _selected_model.layers[backbone_position + 1:]:

        classifier_output = layer(
            classifier_output,
            training=False
        )

    return tf.keras.Model(
        inputs=gradcam_input,
        outputs=[
            feature_maps,
            classifier_output
        ]
    )


try:

    deep_feature_model = create_deep_feature_model(
        model
    )

    gradcam_model = create_gradcam_model(
        model,
        selected_dataset
    )

except Exception as error:

    st.error(
        "Could not prepare the feature-extraction or Grad-CAM model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam_map(
    processed_image,
    class_index,
    selected_gradcam_model
):

    with tf.GradientTape() as tape:

        feature_maps, predictions = selected_gradcam_model(
            processed_image,
            training=False
        )

        target_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        target_score,
        feature_maps
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
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

    return tf.squeeze(
        cam_tensor
    ).numpy()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_map(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    minimum = np.min(values)
    maximum = np.max(values)

    return (
        values - minimum
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

    ax.imshow(image)

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

    map1 = normalize_map(map1).flatten()
    map2 = normalize_map(map2).flatten()

    if np.std(map1) < 1e-12 or np.std(map2) < 1e-12:
        return 0.0

    correlation, _ = pearsonr(
        map1,
        map2
    )

    return float(correlation)


def calculate_ssim(
    map1,
    map2
):

    map1 = normalize_map(map1)
    map2 = normalize_map(map2)

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

    map1 = normalize_map(map1)
    map2 = normalize_map(map2)

    threshold1 = np.percentile(
        map1,
        80
    )

    threshold2 = np.percentile(
        map2,
        80
    )

    mask1 = map1 >= threshold1
    mask2 = map2 >= threshold2

    intersection = np.logical_and(
        mask1,
        mask2
    ).sum()

    union = np.logical_or(
        mask1,
        mask2
    ).sum()

    return float(
        intersection / (union + 1e-8)
    )


# ============================================================
# ZFMIS FEATURE NAME VALIDATION
# ============================================================

selected_feature_names = ZFMIS_FEATURES[
    selected_dataset
]

selected_feature_names = [
    FEATURE_ALIASES.get(name, name)
    for name in selected_feature_names
]

missing_features = [
    name
    for name in selected_feature_names
    if name not in FEATURE_NAMES
]

if missing_features:

    st.error(
        "The selected ZFMIS configuration contains unknown features: "
        f"{missing_features}"
    )

    st.stop()# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    f"Upload a {DATASET_MODALITIES[selected_dataset]} image",
    type=["png", "jpg", "jpeg"],
    help=(
        "Upload an image belonging to the selected dataset. "
        "The model was trained for its corresponding image modality."
    ),
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        ## How to use

        1. Select one of the five medical image datasets in the sidebar.
        2. Upload an image belonging to that dataset.
        3. View the predicted class, confidence and class probabilities.
        4. Inspect Grad-CAM and SFMOV visual explanations.
        5. Review the 128-D deep representation and 26 statistical features.
        6. Inspect the dataset-specific ZFMIS feature subset.
        7. Review the available Rule-Fidelity experiment information.
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception as error:

    st.error("The uploaded file could not be read.")

    st.exception(error)

    st.stop()


# ============================================================
# INPUT IMAGE
# ============================================================

st.subheader("1. Input Medical Image")

col1, col2 = st.columns([1, 2])

with col1:

    st.image(
        image,
        caption=f"Uploaded {DATASET_MODALITIES[selected_dataset]} image",
        use_container_width=True,
    )

with col2:

    st.markdown(
        """
        **Image successfully loaded.**

        The image is converted to RGB, resized to **224 × 224**, and
        processed using the MobileNetV2 preprocessing function.
        """
    )

processed_image = preprocess_input_image(image)


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

try:

    (
        predicted_index,
        predicted_class,
        confidence,
        probabilities,
    ) = predict_image(model, processed_image, class_names)

except Exception as error:

    st.error("Prediction failed.")

    st.exception(error)

    st.stop()


# ============================================================
# PREDICTION DISPLAY
# ============================================================

st.subheader("2. Deep Learning Prediction")

col1, col2 = st.columns(2)

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
        "Class": class_names,
        "Probability (%)": probabilities * 100,
    }
)

st.dataframe(
    probability_df.style.format(
        {"Probability (%)": "{:.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(
    probability_df.set_index("Class")
)


# ============================================================
# GRAD-CAM
# ============================================================

st.subheader("3. Grad-CAM Explanation")

try:

    gradcam_map = generate_gradcam_map(
        processed_image,
        predicted_index,
        gradcam_model,
    )

except Exception as error:

    st.error("Grad-CAM generation failed for this image.")

    st.exception(error)

    st.stop()


gradcam_overlay = create_overlay(
    image,
    gradcam_map
)

col1, col2, col3 = st.columns(3)

with col1:

    st.image(
        image,
        caption="Original Image",
        use_container_width=True,
    )

with col2:

    st.image(
        gradcam_map,
        caption="Grad-CAM Heatmap",
        use_container_width=True,
    )

with col3:

    st.pyplot(
        gradcam_overlay,
        clear_figure=True,
    )


# ============================================================
# ORIGINAL SFMOV
# ============================================================

st.subheader("4. Original SFMOV Explanation")

feature_map_output, _ = gradcam_model(
    processed_image,
    training=False,
)

feature_maps = feature_map_output.numpy()

# Preserve the original SFMOV implementation and equal-weight formula.
original_sfmov = create_original_sfmov(
    feature_maps
)

original_sfmov = normalize_map(
    original_sfmov
)

sfmov_tensor = tf.convert_to_tensor(
    original_sfmov[..., np.newaxis],
    dtype=tf.float32,
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

col1, col2, col3 = st.columns(3)

with col1:

    st.image(
        image,
        caption="Original Image",
        use_container_width=True,
    )

with col2:

    st.image(
        original_sfmov,
        caption="Original SFMOV Heatmap",
        use_container_width=True,
    )

with col3:

    st.pyplot(
        sfmov_overlay,
        clear_figure=True,
    )

st.caption(
    "Original SFMOV combines normalized mean, skewness and entropy "
    "maps using equal weighting. The original SFMOV implementation "
    "has been retained."
)


# ============================================================
# GRAD-CAM VS SFMOV
# ============================================================

st.subheader("5. Grad-CAM vs SFMOV Comparison")

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

col1, col2, col3 = st.columns(3)

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
    "These metrics measure similarity between explanation maps. "
    "Similarity alone does not establish explanation quality."
)


# ============================================================
# 128-D DEEP FEATURES
# ============================================================

st.subheader("6. Deep Feature Representation")

deep_features = deep_feature_model.predict(
    processed_image,
    verbose=0,
)

deep_features = np.asarray(
    deep_features,
    dtype=np.float32,
).reshape(-1)

if len(deep_features) != 128:

    st.error(
        f"Expected 128 deep features, but extracted {len(deep_features)}."
    )

    st.stop()

st.write(
    f"Extracted deep features: **{len(deep_features)}**"
)

with st.expander("View all 128 deep feature values"):

    deep_feature_df = pd.DataFrame(
        {
            "Feature Index": np.arange(
                1,
                len(deep_features) + 1
            ),
            "Value": deep_features,
        }
    )

    st.dataframe(
        deep_feature_df.style.format(
            {"Value": "{:.6f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 26 STATISTICAL FEATURES
# ============================================================

st.subheader("7. Statistical Feature Extraction")

statistical_values = calculate_features(
    deep_features
)

statistical_values = np.asarray(
    statistical_values,
    dtype=np.float32,
).reshape(-1)

if len(statistical_values) != 26:

    st.error(
        f"Expected 26 statistical features, but received "
        f"{len(statistical_values)}."
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
        "Value": statistical_values,
    }
)

st.write(
    f"Extracted statistical features: **{len(statistical_values)}**"
)

st.dataframe(
    statistical_df.style.format(
        {"Value": "{:.6f}"}
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ZFMIS FEATURE SELECTION
# ============================================================

st.subheader("8. ZFMIS Feature Selection")

selected_values_1d = np.asarray(
    [
        statistical_dict[name]
        for name in selected_feature_names
    ],
    dtype=np.float32,
)

selected_values = selected_values_1d.reshape(
    1,
    -1
)

zmfis_df = pd.DataFrame(
    {
        "Selected Feature": selected_feature_names,
        "Value": selected_values_1d,
    }
)

st.write(
    f"ZFMIS selected **{len(selected_feature_names)} features** "
    "from the original 26 statistical features."
)

st.dataframe(
    zmfis_df.style.format(
        {"Value": "{:.6f}"}
    ),
    use_container_width=True,
    hide_index=True,
)# ============================================================
# RULE-FIDELITY EXPLANATION
# ============================================================

st.subheader("9. Rule-Fidelity Explanation")

st.markdown(
    """
    Rule-Fidelity evaluates how closely a decision-tree surrogate
    reproduces the predictions of the original deep learning model.

    The saved fidelity values below are dataset-level experimental
    results. They are not calculated from the currently uploaded image.
    """
)

saved_fidelity = SAVED_RULE_FIDELITY.get(
    selected_dataset
)

if saved_fidelity is not None:

    st.metric(
        "Saved Dataset-Level Rule-Fidelity",
        f"{saved_fidelity * 100:.2f}%"
    )

st.caption(
    "Fidelity measures agreement with the deep learning model, "
    "not classification accuracy against ground-truth labels."
)


# ============================================================
# LIVE COVID RULE-FIDELITY
# ============================================================

if selected_dataset == "covid19":

    st.markdown("#### Live COVID Decision-Tree Surrogate")

    if os.path.exists(COVID_RULE_TREE_PATH):

        try:

            with open(
                COVID_RULE_TREE_PATH,
                "rb"
            ) as file:

                rule_tree = pickle.load(file)

            covid_selected_features = ZFMIS_FEATURES[
                "covid19"
            ]

            covid_selected_features = [
                FEATURE_ALIASES.get(name, name)
                for name in covid_selected_features
            ]

            covid_rule_values = np.asarray(
                [
                    statistical_dict[name]
                    for name in covid_selected_features
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            surrogate_prediction = int(
                rule_tree.predict(
                    covid_rule_values
                )[0]
            )

            if 0 <= surrogate_prediction < len(class_names):

                surrogate_class = class_names[
                    surrogate_prediction
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Deep Learning Prediction",
                        predicted_class
                    )

                with col2:

                    st.metric(
                        "Decision Tree Prediction",
                        surrogate_class
                    )

                if surrogate_class == predicted_class:

                    st.success(
                        "The decision-tree surrogate agrees "
                        "with the deep learning prediction."
                    )

                else:

                    st.warning(
                        "The decision-tree surrogate disagrees "
                        "with the deep learning prediction."
                    )

                st.write(
                    "**Live surrogate prediction:**",
                    surrogate_class
                )

            else:

                st.warning(
                    "The surrogate returned a class index "
                    "outside the configured class range."
                )

        except Exception as error:

            st.error(
                "The COVID Rule-Fidelity model could not be used "
                "for live surrogate prediction."
            )

            st.exception(error)

    else:

        st.warning(
            "The serialized COVID Rule-Fidelity decision-tree "
            "model was not found."
        )

else:

    st.info(
        "Live per-image Rule-Fidelity prediction is not enabled "
        "for this dataset. The serialized surrogate inference "
        "pipeline, including its exact feature scaling and ordering, "
        "has not been verified. The saved dataset-level fidelity "
        "result is shown for experimental reference only."
    )


# ============================================================
# COMPLETE PIPELINE SUMMARY
# ============================================================

st.subheader("10. Complete Explainable AI Pipeline")

pipeline_steps = [
    "Medical image upload",
    "RGB conversion and resizing to 224 × 224",
    "MobileNetV2 image preprocessing",
    "Dataset-specific MobileNetV2 classification",
    "Prediction and class probability calculation",
    "Grad-CAM heatmap generation",
    "Original equal-weight SFMOV explanation",
    "Grad-CAM and SFMOV comparison",
    "128-dimensional deep feature extraction",
    "26 statistical feature calculations",
    f"ZFMIS selection of {len(selected_feature_names)} features",
    "Rule-Fidelity experimental analysis",
]

for index, step in enumerate(
    pipeline_steps,
    start=1
):

    st.write(
        f"**{index}.** {step}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

st.divider()

st.header("Final Prediction Report")

st.markdown(
    f"""
    ### Dataset Information

    - **Dataset:** {selected_dataset_label}
    - **Image modality:** {DATASET_MODALITIES[selected_dataset]}
    - **Model:** MobileNetV2
    - **Input dimensions:** 224 × 224 × 3

    ### Classification Result

    - **Predicted class:** {predicted_class}
    - **Confidence:** {confidence * 100:.2f}%
    - **Number of classes:** {len(class_names)}

    ### Explainability Results

    - **Grad-CAM:** Generated
    - **SFMOV:** Generated using the original equal-weight method
    - **Deep features:** {len(deep_features)}
    - **Statistical features:** {len(statistical_values)}
    - **ZFMIS selected features:** {len(selected_feature_names)}

    ### Explanation Comparison

    - **Pearson correlation:** {pearson_value:.4f}
    - **SSIM:** {ssim_value:.4f}
    - **Top-20% IoU:** {iou_value:.4f}

    ### Rule-Fidelity

    - **Saved dataset-level fidelity:** {saved_fidelity * 100:.2f}%
      (not an image-specific measurement)
    """
)

st.divider()

st.caption(
    "This project is an academic research prototype. Model predictions "
    "and visual explanations are not clinically validated. They should "
    "not be used for medical diagnosis or treatment decisions."
)

st.success(
    "Medical image analysis and explainability report completed."
)