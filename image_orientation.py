from __future__ import annotations

from PIL import Image, ImageOps


EXIF_ORIENTATION_TAG = 274


def read_exif_orientation(image: Image.Image) -> int:
    """Return the source EXIF orientation, defaulting to normal orientation."""
    try:
        return int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except (TypeError, ValueError):
        return 1


def normalize_uploaded_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once and return independent RGB pixel data.

    The returned image is intended to be saved as working.png. PNG does not
    carry the original EXIF orientation, so downstream rotation, detection,
    and crop coordinates all use the same physical pixel coordinate system.
    """
    normalized = ImageOps.exif_transpose(image)
    exif = normalized.getexif()
    if EXIF_ORIENTATION_TAG in exif:
        del exif[EXIF_ORIENTATION_TAG]
    normalized.info.pop("exif", None)
    return normalized.convert("RGB")
