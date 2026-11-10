
import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
DEFAULT_STEPS = 20
DEFAULT_RANDOM_RUNS = 3
BLUR_RADIUS = 12


# --------------------------------------------------
# CREATE IMAGE PATCHES
# --------------------------------------------------

def create_image_patches(image, patch_size=PATCH_SIZE):

    height, width = image.shape[:2]

    patches = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)

            patches.append((y, y_end, x, x_end))

    return patches


# --------------------------------------------------
# RANK PATCHES USING EXPLANATION MAP
# --------------------------------------------------

def rank_patches(explanation_map, patches):

    scores = []

    for y1, y2, x1, x2 in patches:

        patch_score = np.mean(
            explanation_map[y1:y2, x1:x2]
        )

        scores.append(patch_score)

    return np.argsort(scores)[::-1]


# --------------------------------------------------
# GENERATE DELETION AND INSERTION IMAGES
# --------------------------------------------------

def generate_perturbation_sequences(
    original_image,
    baseline_image,
    patches,
    patch_order,
    steps=DEFAULT_STEPS
):

    deletion_images = [original_image.copy()]
    insertion_images = [baseline_image.copy()]

    deletion_current = original_image.copy()
    insertion_current = baseline_image.copy()

    patch_groups = np.array_split(
        patch_order,
        steps
    )

    for group in patch_groups:

        for patch_index in group:

            y1, y2, x1, x2 = patches[int(patch_index)]

            # Deletion: replace important regions with baseline.
            deletion_current[y1:y2, x1:x2] = (
                baseline_image[y1:y2, x1:x2]
            )

            # Insertion: restore important regions.
            insertion_current[y1:y2, x1:x2] = (
                original_image[y1:y2, x1:x2]
            )

        deletion_images.append(
            deletion_current.copy()
        )

        insertion_images.append(
            insertion_current.copy()
        )

    return deletion_images, insertion_images


# --------------------------------------------------
# PREPROCESS AND PREDICT PERTURBED IMAGES
# --------------------------------------------------

def predict_perturbed_images(
    model,
    images,
    preprocess_fn,
    target_class_index,
    batch_size=16
):

    processed_images = []

    for image_array in images:

        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            image_array
        )

        processed = preprocess_fn(
            pil_image
        )

        processed = np.asarray(
            processed,
            dtype=np.float32
        )

        if processed.shape[0] != 1:
            raise ValueError(
                "Preprocessing function must return "
                "one image per call."
            )

        processed_images.append(
            processed[0]
        )

    input_batch = np.stack(
        processed_images,
        axis=0
    )

    predictions = model.predict(
        input_batch,
        batch_size=batch_size,
        verbose=0
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    if predictions.ndim != 2:
        raise ValueError(
            "Expected model predictions with shape "
            "(number_of_images, number_of_classes)."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model returned invalid prediction values."
        )

    if not 0 <= target_class_index < predictions.shape[1]:
        raise ValueError(
            "Target class index is outside the model output range."
        )

    return predictions[:, target_class_index]


# --------------------------------------------------
# CALCULATE AREA UNDER THE CURVE
# --------------------------------------------------

def calculate_auc(curve):

    curve = np.asarray(
        curve,
        dtype=np.float64
    )

    fractions = np.linspace(
        0.0,
        1.0,
        len(curve)
    )

    return float(
        np.trapezoid(
            curve,
            fractions
        )
    )


# --------------------------------------------------
# MAIN FAITHFULNESS EVALUATION
# --------------------------------------------------

def evaluate_faithfulness(
    model,
    original_image,
    explanation_map,
    target_class_index,
    preprocess_fn,
    steps=DEFAULT_STEPS,
    random_runs=DEFAULT_RANDOM_RUNS,
    random_state=42
):

    # Validate configuration.

    if steps < 2:
        raise ValueError(
            "At least two evaluation steps are required."
        )

    if random_runs < 1:
        raise ValueError(
            "At least one random baseline run is required."
        )

    # Resize original image exactly as the application does.

    resized_image = original_image.convert(
        "RGB"
    ).resize(IMAGE_SIZE)

    original_array = np.asarray(
        resized_image,
        dtype=np.uint8
    )

    # Validate explanation map.

    explanation_map = np.asarray(
        explanation_map,
        dtype=np.float32
    )

    if explanation_map.shape != (
        IMAGE_SIZE[1],
        IMAGE_SIZE[0]
    ):
        raise ValueError(
            "Explanation map must have shape 224 x 224."
        )

    if not np.all(np.isfinite(explanation_map)):
        raise ValueError(
            "Explanation map contains invalid values."
        )

    explanation_map = np.maximum(
        explanation_map,
        0
    )

    if np.max(explanation_map) - np.min(explanation_map) < 1e-8:
        raise ValueError(
            "Explanation map has no meaningful variation."
        )

    # Create blurred baseline image.

    baseline_pil = resized_image.filter(
        ImageFilter.GaussianBlur(
            radius=BLUR_RADIUS
        )
    )

    baseline_array = np.asarray(
        baseline_pil,
        dtype=np.uint8
    )

    # Divide image into patches.

    patches = create_image_patches(
        original_array
    )

    # Rank patches according to Grad-CAM.

    important_order = rank_patches(
        explanation_map,
        patches
    )

    # Generate Grad-CAM deletion and insertion sequences.

    deletion_images, insertion_images = (
        generate_perturbation_sequences(
            original_array,
            baseline_array,
            patches,
            important_order,
            steps
        )
    )

    # Store all sequences for batched prediction.

    sequences = [
        ("deletion", deletion_images),
        ("insertion", insertion_images)
    ]

    rng = np.random.default_rng(
        random_state
    )

    random_orders = []

    for _ in range(random_runs):

        random_order = rng.permutation(
            len(patches)
        )

        random_orders.append(
            random_order
        )

        random_deletion, random_insertion = (
            generate_perturbation_sequences(
                original_array,
                baseline_array,
                patches,
                random_order,
                steps
            )
        )

        sequences.append(
            ("random_deletion", random_deletion)
        )

        sequences.append(
            ("random_insertion", random_insertion)
        )

    # Combine all images into one prediction batch.

    all_images = []
    sequence_ranges = {}

    for sequence_index, (name, images) in enumerate(sequences):

        start = len(all_images)

        all_images.extend(images)

        end = len(all_images)

        sequence_ranges[
            (name, sequence_index)
        ] = (start, end)

    all_scores = predict_perturbed_images(
        model=model,
        images=all_images,
        preprocess_fn=preprocess_fn,
        target_class_index=target_class_index
    )

    # Extract individual prediction curves.

    curves = {}

    for sequence_index, (name, _) in enumerate(sequences):

        start, end = sequence_ranges[
            (name, sequence_index)
        ]

        curves[
            (name, sequence_index)
        ] = all_scores[start:end].tolist()

    deletion_curve = np.asarray(
        curves[("deletion", 0)]
    )

    insertion_curve = np.asarray(
        curves[("insertion", 1)]
    )

    random_deletion_curves = []
    random_insertion_curves = []

    for run_index in range(random_runs):

        deletion_index = 2 + (run_index * 2)
        insertion_index = deletion_index + 1

        random_deletion_curves.append(
            curves[
                ("random_deletion", deletion_index)
            ]
        )

        random_insertion_curves.append(
            curves[
                ("random_insertion", insertion_index)
            ]
        )

    random_deletion_curves = np.asarray(
        random_deletion_curves,
        dtype=np.float64
    )

    random_insertion_curves = np.asarray(
        random_insertion_curves,
        dtype=np.float64
    )

    mean_random_deletion = np.mean(
        random_deletion_curves,
        axis=0
    )

    mean_random_insertion = np.mean(
        random_insertion_curves,
        axis=0
    )

    # Calculate AUC values.

    deletion_auc = calculate_auc(
        deletion_curve
    )

    insertion_auc = calculate_auc(
        insertion_curve
    )

    random_deletion_aucs = [
        calculate_auc(curve)
        for curve in random_deletion_curves
    ]

    random_insertion_aucs = [
        calculate_auc(curve)
        for curve in random_insertion_curves
    ]

    random_deletion_auc = float(
        np.mean(random_deletion_aucs)
    )

    random_insertion_auc = float(
        np.mean(random_insertion_aucs)
    )

    # Positive values indicate improvement over random ordering.

    deletion_advantage = (
        random_deletion_auc - deletion_auc
    )

    insertion_advantage = (
        insertion_auc - random_insertion_auc
    )

    return {

        "deletion_auc": deletion_auc,

        "insertion_auc": insertion_auc,

        "random_deletion_auc": random_deletion_auc,

        "random_insertion_auc": random_insertion_auc,

        "deletion_advantage": float(
            deletion_advantage
        ),

        "insertion_advantage": float(
            insertion_advantage
        ),

        "deletion_curve": deletion_curve.tolist(),

        "insertion_curve": insertion_curve.tolist(),

        "random_deletion_curve": mean_random_deletion.tolist(),

        "random_insertion_curve": mean_random_insertion.tolist(),

        "random_deletion_aucs": random_deletion_aucs,

        "random_insertion_aucs": random_insertion_aucs,

        "random_runs": random_runs,

        "steps": steps,

        "target_class_index": int(
            target_class_index
        )
    }
