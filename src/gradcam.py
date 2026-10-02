import tensorflow as tf
import numpy as np


def make_gradcam_heatmap(
    image_array,
    model,
    last_conv_layer_name="out_relu",
    pred_index=None
):
    """
    Generate a Grad-CAM heatmap for a single input image.

    Parameters
    ----------
    image_array : numpy.ndarray
        Preprocessed image with shape (1, 224, 224, 3).

    model : tf.keras.Model
        Trained classification model.

    last_conv_layer_name : str
        Convolutional feature layer used for Grad-CAM.

    pred_index : int or None
        Target class index. If None, uses the predicted class.

    Returns
    -------
    heatmap : numpy.ndarray
        Normalized Grad-CAM heatmap with values in [0, 1].

    predicted_index : int
        Class index used for Grad-CAM.
    """

    # Create a model that returns:
    # 1. Feature maps from the selected convolutional layer
    # 2. Final model predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array,
            training=False
        )

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[:, pred_index]

    # Gradient of class score with respect to feature maps
    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weight feature maps by the importance of each channel
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU: retain positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize to [0, 1]
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy(), int(pred_index)