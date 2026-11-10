
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity


def calculate_heatmap_iou(map1, map2, percentile=80):
    """Calculate IoU between the top percentile regions of two heatmaps."""

    map1 = np.asarray(map1, dtype=np.float32)
    map2 = np.asarray(map2, dtype=np.float32)

    threshold1 = np.percentile(map1, percentile)
    threshold2 = np.percentile(map2, percentile)

    mask1 = map1 >= threshold1
    mask2 = map2 >= threshold2

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return float(intersection / union) if union > 0 else 1.0


def compare_heatmaps(original_map, perturbed_map):
    """Compare two explanation heatmaps using three metrics."""

    original_map = np.asarray(original_map, dtype=np.float32)
    perturbed_map = np.asarray(perturbed_map, dtype=np.float32)

    if original_map.shape != perturbed_map.shape:
        raise ValueError("Both heatmaps must have the same dimensions.")

    original_map = np.nan_to_num(original_map)
    perturbed_map = np.nan_to_num(perturbed_map)

    if np.std(original_map) == 0 or np.std(perturbed_map) == 0:
        correlation = 0.0
    else:
        correlation = float(
            pearsonr(original_map.ravel(), perturbed_map.ravel())[0]
        )

    ssim = float(
        structural_similarity(
            original_map,
            perturbed_map,
            data_range=1.0
        )
    )

    iou = calculate_heatmap_iou(
        original_map,
        perturbed_map,
        percentile=80
    )

    return {
        "pearson_correlation": correlation,
        "ssim": ssim,
        "top_20_iou": iou
    }


def create_perturbed_images(image, num_perturbations=5):
    """Generate mildly perturbed versions of an input image."""

    image = image.convert("RGB").resize((224, 224))
    perturbed_images = []

    for i in range(num_perturbations):

        if i == 0:
            perturbed_images.append(image.copy())

        elif i == 1:
            perturbed_images.append(
                image.filter(ImageFilter.GaussianBlur(radius=0.5))
            )

        elif i == 2:
            perturbed_images.append(
                image.filter(ImageFilter.GaussianBlur(radius=1.0))
            )

        elif i == 3:
            perturbed_images.append(
                Image.eval(image, lambda x: min(255, int(x * 1.05)))
            )

        elif i == 4:
            perturbed_images.append(
                Image.eval(image, lambda x: max(0, int(x * 0.95)))
            )

    return perturbed_images


def evaluate_stability(
    image,
    generate_explanation,
    num_perturbations=5
):
    """
    Evaluate Grad-CAM explanation stability under mild
    image perturbations.

    generate_explanation must accept a PIL image and return
    a 2D normalized explanation heatmap.
    """

    images = create_perturbed_images(
        image,
        num_perturbations=num_perturbations
    )

    original_map = generate_explanation(images[0])

    original_map = np.asarray(original_map, dtype=np.float32)

    if original_map.ndim != 2:
        raise ValueError("Explanation function must return a 2D heatmap.")

    original_map = np.nan_to_num(original_map)
    original_map = np.clip(original_map, 0, 1)

    results = []

    for index, perturbed_image in enumerate(images[1:], start=1):

        perturbed_map = generate_explanation(perturbed_image)

        perturbed_map = np.asarray(
            perturbed_map,
            dtype=np.float32
        )

        if perturbed_map.shape != original_map.shape:
            raise ValueError("Generated heatmaps must have matching shapes.")

        perturbed_map = np.nan_to_num(perturbed_map)
        perturbed_map = np.clip(perturbed_map, 0, 1)

        metrics = compare_heatmaps(
            original_map,
            perturbed_map
        )

        metrics["perturbation"] = index
        results.append(metrics)

    if not results:
        raise ValueError("At least two images are required.")

    summary = {
        "mean_pearson": float(np.mean([
            r["pearson_correlation"] for r in results
        ])),
        "mean_ssim": float(np.mean([
            r["ssim"] for r in results
        ])),
        "mean_top_20_iou": float(np.mean([
            r["top_20_iou"] for r in results
        ]))
    }

    return {
        "individual_results": results,
        "summary": summary
    }
