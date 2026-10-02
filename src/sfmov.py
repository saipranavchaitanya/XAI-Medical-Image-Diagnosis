
import numpy as np
import tensorflow as tf


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


def calculate_entropy_map(feature_maps):
    """Calculate spatial entropy map from feature activations."""

    feature_maps = np.asarray(
        feature_maps,
        dtype=np.float32
    )

    probabilities = np.abs(
        feature_maps
    ) + 1e-8

    probabilities = probabilities / (
        np.sum(
            probabilities,
            axis=-1,
            keepdims=True
        ) + 1e-8
    )

    entropy_map = -np.sum(
        probabilities * np.log(
            probabilities
        ),
        axis=-1
    )

    return entropy_map


def calculate_skewness_map(feature_maps):
    """Calculate spatial skewness map across channels."""

    x = np.asarray(
        feature_maps,
        dtype=np.float32
    )

    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    std = np.std(
        x,
        axis=-1,
        keepdims=True
    )

    skewness = np.mean(
        (
            (x - mean) /
            (std + 1e-8)
        ) ** 3,
        axis=-1
    )

    return skewness


def calculate_mean_map(feature_maps):
    """Calculate spatial mean activation map."""

    return np.mean(
        feature_maps,
        axis=-1
    )


def create_original_sfmov(feature_maps):
    """Create the original equal-weight SFMOV map."""

    mean_map = calculate_mean_map(
        feature_maps
    )

    skewness_map = calculate_skewness_map(
        feature_maps
    )

    entropy_map = calculate_entropy_map(
        feature_maps
    )

    mean_map = normalize_map(
        mean_map
    )

    skewness_map = normalize_map(
        skewness_map
    )

    entropy_map = normalize_map(
        entropy_map
    )

    sfmov = (
        mean_map +
        skewness_map +
        entropy_map
    ) / 3.0

    return normalize_map(
        sfmov
    )


def upscale_map(
    explanation_map,
    target_size=(224, 224)
):
    explanation_map = tf.convert_to_tensor(
        explanation_map[..., np.newaxis],
        dtype=tf.float32
    )

    result = tf.image.resize(
        explanation_map,
        target_size
    )

    return tf.squeeze(
        result
    ).numpy()
