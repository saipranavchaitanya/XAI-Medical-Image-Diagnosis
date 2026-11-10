
import numpy as np
from PIL import Image, ImageFilter


def create_region_masks(explanation_map, percentile=80):
    """
    Create equal-sized salient and non-salient masks.

    Salient: highest explanation values.
    Non-salient: lowest explanation values.
    Both masks contain exactly the same number of pixels.
    """

    explanation_map = np.asarray(
        explanation_map,
        dtype=np.float32
    )

    if explanation_map.ndim != 2:
        raise ValueError("Explanation map must be 2D.")

    if not np.all(np.isfinite(explanation_map)):
        raise ValueError("Explanation map contains invalid values.")

    if not 50 <= percentile < 100:
        raise ValueError("Percentile must be between 50 and 100.")

    flat_map = explanation_map.flatten()
    total_pixels = flat_map.size

    region_size = int(
        round(total_pixels * (100 - percentile) / 100)
    )

    region_size = max(1, region_size)

    sorted_indices = np.argsort(
        flat_map,
        kind="stable"
    )

    # Bottom explanation values
    non_salient_indices = sorted_indices[:region_size]

    # Top explanation values
    salient_indices = sorted_indices[-region_size:]

    salient_mask = np.zeros(total_pixels, dtype=bool)
    non_salient_mask = np.zeros(total_pixels, dtype=bool)

    salient_mask[salient_indices] = True
    non_salient_mask[non_salient_indices] = True

    return (
        salient_mask.reshape(explanation_map.shape),
        non_salient_mask.reshape(explanation_map.shape)
    )


def perturb_region(image, mask, replacement="blur"):
    """
    Replace the selected image region with a blurred baseline.
    """

    image = image.convert("RGB").resize((224, 224))

    original = np.asarray(
        image,
        dtype=np.float32
    )

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != original.shape[:2]:
        raise ValueError(
            "Explanation mask must match image dimensions."
        )

    if replacement == "blur":
        baseline = np.asarray(
            image.filter(
                ImageFilter.GaussianBlur(radius=10)
            ),
            dtype=np.float32
        )
    else:
        raise ValueError("Unsupported replacement method.")

    perturbed = original.copy()
    perturbed[mask] = baseline[mask]

    return Image.fromarray(
        np.clip(perturbed, 0, 255).astype(np.uint8)
    )


def evaluate_sensitivity(
    image,
    explanation_map,
    predict_function,
    target_class_index,
    percentile=80
):
    """
    Evaluate model sensitivity to equal-sized salient
    and non-salient image regions.

    predict_function must accept a PIL image and return
    a one-dimensional class probability array.
    """

    image = image.convert("RGB").resize((224, 224))

    explanation_map = np.asarray(
        explanation_map,
        dtype=np.float32
    )

    if explanation_map.shape != (224, 224):
        raise ValueError(
            "Explanation map must have shape (224, 224)."
        )

    if not np.all(np.isfinite(explanation_map)):
        raise ValueError(
            "Explanation map contains invalid values."
        )

    salient_mask, non_salient_mask = create_region_masks(
        explanation_map,
        percentile
    )

    original_probabilities = np.asarray(
        predict_function(image)
    ).flatten()

    if not 0 <= target_class_index < len(original_probabilities):
        raise ValueError("Invalid target class index.")

    original_confidence = float(
        original_probabilities[target_class_index]
    )

    salient_image = perturb_region(
        image,
        salient_mask
    )

    non_salient_image = perturb_region(
        image,
        non_salient_mask
    )

    salient_probabilities = np.asarray(
        predict_function(salient_image)
    ).flatten()

    non_salient_probabilities = np.asarray(
        predict_function(non_salient_image)
    ).flatten()

    salient_confidence = float(
        salient_probabilities[target_class_index]
    )

    non_salient_confidence = float(
        non_salient_probabilities[target_class_index]
    )

    salient_drop = (
        original_confidence - salient_confidence
    )

    non_salient_drop = (
        original_confidence - non_salient_confidence
    )

    return {
        "original_confidence": original_confidence,
        "salient_confidence": salient_confidence,
        "non_salient_confidence": non_salient_confidence,

        "salient_confidence_drop": float(salient_drop),

        "non_salient_confidence_drop": float(non_salient_drop),

        "sensitivity_difference": float(
            salient_drop - non_salient_drop
        ),

        "salient_region_percentage": float(
            salient_mask.mean() * 100
        ),

        "non_salient_region_percentage": float(
            non_salient_mask.mean() * 100
        )
    }
