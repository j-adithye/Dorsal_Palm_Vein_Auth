
#!/usr/bin/env python3
"""
augmentation.py
===============
Augmentation pipeline for palm vein recognition.

Since lighting is controlled (fixed NIR LEDs), brightness/contrast ranges
are reduced. New augmentations added for better structural understanding:
  - Random erasing   : forces model not to rely on any single vein segment
  - Sharpen/blur     : simulates focus and pressure variation between captures

All augmentations are probabilistic — not every transform applies every time.
Horizontal flip deliberately excluded — left/right vein patterns are not symmetric.

Usage:
    from augmentation import augment_image
    augmented = augment_image(img)   # float32 (H, W, 1) -> (H, W, 1)
"""

import random
import numpy as np
import cv2


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

AUG_CONFIG = {
    # Spatial
    "rotation_deg":       12,     # ± degrees
    "translate_frac":     0.05,   # ± fraction of image size

    # Photometric — reduced since lighting is controlled

    "noise_std":          6,      # gaussian noise std in pixel space (was 8)

    # Random erasing
    "erase_prob":         0.7,    # probability of applying
    "erase_min_frac":     0.02,   # min erased area as fraction of image
    "erase_max_frac":     0.12,   # max erased area as fraction of image
    "erase_max_patches":  6,      # max number of patches per image


    # Sharpen / blur
    "sharpen_blur_prob":  0.4,    # probability of applying
    "blur_max_kernel":    3,      # max gaussian blur kernel radius (odd only)
    "sharpen_strength":   0.4,    # strength of unsharp mask
}


# ─────────────────────────────────────────────
#  SPATIAL TRANSFORMS
# ─────────────────────────────────────────────

def _rotate(img):
    h, w  = img.shape
    angle = random.uniform(-AUG_CONFIG["rotation_deg"], AUG_CONFIG["rotation_deg"])
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _translate(img):
    h, w = img.shape
    frac = AUG_CONFIG["translate_frac"]
    tx   = random.uniform(-frac, frac) * w
    ty   = random.uniform(-frac, frac) * h
    M    = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)





# ─────────────────────────────────────────────
#  PHOTOMETRIC TRANSFORMS
# ─────────────────────────────────────────────



def _gaussian_noise(img):
    """Mild sensor noise."""
    std   = AUG_CONFIG["noise_std"] / 255.0
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _sharpen_blur(img):
    """
    Randomly either sharpens or blurs the image.
    Sharpening: simulates high-pressure firm hand placement (clearer veins).
    Blurring   : simulates slight motion or soft placement (softer veins).
    """
    if random.random() > AUG_CONFIG["sharpen_blur_prob"]:
        return img

    if random.random() < 0.5:
        # Blur
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # Sharpen via unsharp mask
        strength = AUG_CONFIG["sharpen_strength"]
        blurred  = cv2.GaussianBlur(img, (5, 5), 0)
        sharp    = img + strength * (img - blurred)
        return np.clip(sharp, 0.0, 1.0)


# ─────────────────────────────────────────────
#  RANDOM ERASING
# ─────────────────────────────────────────────

def _random_erase(img):
    """
    Randomly blacks out small rectangular patches.
    Forces model not to rely on any single vein segment for identification.
    Directly addresses over-reliance on prominent veins that vary with hand position.
    """
    if random.random() > AUG_CONFIG["erase_prob"]:
        return img

    img    = img.copy()
    h, w   = img.shape
    area   = h * w
    n_patches = random.randint(1, AUG_CONFIG["erase_max_patches"])

    for _ in range(n_patches):
        erase_area = random.uniform(
            AUG_CONFIG["erase_min_frac"],
            AUG_CONFIG["erase_max_frac"]
        ) * area
        aspect = random.uniform(0.3, 3.0)
        ph     = int(np.sqrt(erase_area / aspect))
        pw     = int(np.sqrt(erase_area * aspect))
        ph     = max(1, min(ph, h - 1))
        pw     = max(1, min(pw, w - 1))
        y0     = random.randint(0, h - ph)
        x0     = random.randint(0, w - pw)
        img[y0:y0+ph, x0:x0+pw] = 0.0

    return img


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def augment_image(img):
    """
    Apply augmentation pipeline to a single vein image.

    Args:
        img: float32 array shape (H, W, 1) or (H, W), values in [0, 1].

    Returns:
        float32 array shape (H, W, 1), values in [0, 1].

    Pipeline order:
        1. Spatial transforms  (rotation, translation, grid distortion)
        2. Photometric transforms (brightness, contrast, noise, sharpen/blur)
        3. Random erasing last  (applied after photometric so erased regions stay black)
    """
    if img.ndim == 3:
        img = img[:, :, 0]

    # Spatial
    img = _rotate(img)
    img = _translate(img)

    # Photometric
    img = _sharpen_blur(img)
    img = _gaussian_noise(img)

    
    # Structural
    img = _random_erase(img)

    return img[:, :, np.newaxis]