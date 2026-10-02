import numpy as np

from scipy.stats import (
    entropy,
    gmean,
    hmean,
    kurtosis,
    skew
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "mean",
    "median",
    "variance",
    "geometric_mean",
    "std",
    "minimum",
    "autocorrelation",
    "iqr",
    "kurtosis",
    "entropy",
    "energy",
    "contrast",
    "mad",
    "shannon_entropy",
    "range",
    "skewness",
    "maximum",
    "snr",
    "percentile_10",
    "percentile_25",
    "percentile_50",
    "percentile_75",
    "harmonic_mean",
    "coefficient_variation",
    "sem",
    "rms"
]


# ============================================================
# SAFE POSITIVE VALUES
# ============================================================

def safe_positive(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    return np.abs(x) + 1e-8


# ============================================================
# CALCULATE 26 STATISTICAL FEATURES
# ============================================================

def calculate_features(feature_map):

    x = np.asarray(
        feature_map,
        dtype=np.float32
    ).flatten()

    # Remove invalid values
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if len(x) == 0:

        return np.zeros(
            len(FEATURE_NAMES),
            dtype=np.float32
        )

    x_positive = safe_positive(
        x
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_value = np.mean(x)

    median_value = np.median(x)

    variance_value = np.var(x)

    std_value = np.std(x)

    minimum_value = np.min(x)

    maximum_value = np.max(x)

    range_value = (
        maximum_value -
        minimum_value
    )

    # --------------------------------------------------------
    # Geometric mean
    # --------------------------------------------------------

    try:

        geometric_mean_value = gmean(
            x_positive
        )

    except Exception:

        geometric_mean_value = 0.0

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

    if len(x) > 1:

        x1 = x[:-1]
        x2 = x[1:]

        std_x1 = np.std(x1)
        std_x2 = np.std(x2)

        if (
            std_x1 > 1e-8
            and
            std_x2 > 1e-8
        ):

            autocorrelation = np.corrcoef(
                x1,
                x2
            )[0, 1]

        else:

            autocorrelation = 0.0

    else:

        autocorrelation = 0.0

    # --------------------------------------------------------
    # Interquartile range
    # --------------------------------------------------------

    q25 = np.percentile(
        x,
        25
    )

    q75 = np.percentile(
        x,
        75
    )

    iqr_value = (
        q75 -
        q25
    )

    # --------------------------------------------------------
    # Kurtosis
    # --------------------------------------------------------

    try:

        kurtosis_value = kurtosis(
            x,
            bias=False
        )

    except Exception:

        kurtosis_value = 0.0

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    try:

        probabilities = np.abs(x)

        probabilities = (
            probabilities /
            (
                np.sum(probabilities)
                + 1e-8
            )
        )

        entropy_value = entropy(
            probabilities
        )

    except Exception:

        entropy_value = 0.0

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy_value = np.sum(
        x ** 2
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast_value = np.mean(
        (
            x -
            mean_value
        ) ** 2
    )

    # --------------------------------------------------------
    # Mean absolute deviation
    # --------------------------------------------------------

    mad_value = np.mean(
        np.abs(
            x -
            median_value
        )
    )

    # --------------------------------------------------------
    # Shannon entropy
    # --------------------------------------------------------

    try:

        histogram, _ = np.histogram(
            x,
            bins=10,
            density=False
        )

        histogram = histogram.astype(
            np.float32
        )

        histogram = (
            histogram /
            (
                np.sum(histogram)
                + 1e-8
            )
        )

        histogram = histogram[
            histogram > 0
        ]

        shannon_entropy_value = (
            -np.sum(
                histogram *
                np.log2(histogram)
            )
        )

    except Exception:

        shannon_entropy_value = 0.0

    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------

    try:

        skewness_value = skew(
            x,
            bias=False
        )

    except Exception:

        skewness_value = 0.0

    # --------------------------------------------------------
    # Signal-to-noise ratio
    # --------------------------------------------------------

    snr_value = (
        mean_value /
        (
            std_value +
            1e-8
        )
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    p10 = np.percentile(
        x,
        10
    )

    p25 = np.percentile(
        x,
        25
    )

    p50 = np.percentile(
        x,
        50
    )

    # FIX:
    # The original file used p75 in the feature list
    # but did not define it.
    p75 = np.percentile(
        x,
        75
    )

    # --------------------------------------------------------
    # Harmonic mean
    # --------------------------------------------------------

    try:

        harmonic_mean_value = hmean(
            x_positive
        )

    except Exception:

        harmonic_mean_value = 0.0

    # --------------------------------------------------------
    # Coefficient of variation
    # --------------------------------------------------------

    coefficient_variation_value = (
        std_value /
        (
            abs(mean_value)
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Standard error of mean
    # --------------------------------------------------------

    sem_value = (
        std_value /
        np.sqrt(
            len(x)
        )
    )

    # --------------------------------------------------------
    # Root mean square
    # --------------------------------------------------------

    rms_value = np.sqrt(
        np.mean(
            x ** 2
        )
    )

    # ========================================================
    # FINAL 26 FEATURES
    # ========================================================

    values = [

        mean_value,

        median_value,

        variance_value,

        geometric_mean_value,

        std_value,

        minimum_value,

        autocorrelation,

        iqr_value,

        kurtosis_value,

        entropy_value,

        energy_value,

        contrast_value,

        mad_value,

        shannon_entropy_value,

        range_value,

        skewness_value,

        maximum_value,

        snr_value,

        p10,

        p25,

        p50,

        p75,

        harmonic_mean_value,

        coefficient_variation_value,

        sem_value,

        rms_value
    ]

    return np.nan_to_num(
        np.asarray(
            values,
            dtype=np.float32
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# FEATURE DICTIONARY
# ============================================================

def calculate_feature_dict(
    feature_map
):

    values = calculate_features(
        feature_map
    )

    return dict(
        zip(
            FEATURE_NAMES,
            values
        )
    )