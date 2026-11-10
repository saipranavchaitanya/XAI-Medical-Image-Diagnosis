import tensorflow as tf
import numpy as np



def make_gradcam_heatmap(
    image_array,
    model,
    last_conv_layer_name="out_relu",
    pred_index=None
):
    """
    Generate Grad-CAM for both nested and non-nested models.
    """

    # Find a nested backbone containing the target layer.
    backbone = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if any(
                sublayer.name == last_conv_layer_name
                for sublayer in layer.layers
            ):
                backbone = layer
                break

    if backbone is not None:
        # Extract intermediate feature maps and backbone output.
        target_layer = backbone.get_layer(last_conv_layer_name)

        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                target_layer.output,
                backbone.output
            ]
        )

        # Build classifier head after the nested backbone.
        backbone_position = model.layers.index(backbone)

        head_layers = model.layers[backbone_position + 1:]

        def classify(features):
            x = features
            for layer in head_layers:
                x = layer(x)
            return x

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = feature_model(
                image_array,
                training=False
            )

            predictions = classify(backbone_features)

            if pred_index is None:
                pred_index = int(
                    tf.argmax(predictions[0]).numpy()
                )

            class_score = predictions[:, pred_index]

    else:
        # Original architecture: target layer at model level.
        target_layer = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                image_array,
                training=False
            )

            if pred_index is None:
                pred_index = int(
                    tf.argmax(predictions[0]).numpy()
                )

            class_score = predictions[:, pred_index]

    # Calculate gradients.
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy(), int(pred_index)
