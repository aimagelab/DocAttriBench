import base64
import copy
from io import BytesIO
from math import floor, sqrt, ceil
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def convert_to_bites(image: Image.Image) -> str:
    """
    Convert a PIL image to bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_str


def blur_bounding_box(image, bbox):
    """
    Blurs a specific region of an image defined by a bounding box.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        bbox (tuple): A tuple defining the bounding box (left, upper, right, lower)
                    of the region to be blurred.

    Returns:
        PIL.Image.Image: A copy of the input image with the specified region blurred.

    Note:
        The function creates a copy of the input image to avoid modifying the original image.
        The bounding box coordinates should be within the dimensions of the input image.
    """
    image_c = copy.copy(image)
    region = image_c.crop(bbox)
    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=10))
    image_c.paste(blurred_region, bbox)

    return image_c


def extract_patches(
    image: Image.Image, num_patches: int = 3, overlap: float = 0.3, direction=0
):
    """
    Divide an image into patches with specified overlap.

    Parameters:
    - image: PIL.Image.Image object
    - num_patches: Number of patches to create (default is 3)
    - overlap: Fractional overlap between patches (0.0 to <1.0)

    Returns:
    - List of PIL.Image.Image patches
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")

    width, height = image.size
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale image
        img_array = img_array[:, :, np.newaxis]  # Add channel dim for consistency

    all_patches = []
    all_coords = []

    if direction == 0:  # path horizontally
        patch_dim = width / (
            1 + (num_patches - 1) * (1 - overlap)
        )  # The patch dimension
        step = patch_dim * (
            1 - overlap
        )  # The step, i.e. how much to move to initiate the next patch

        for i in range(num_patches):
            x0 = max(0, int(i * step))
            x1 = min(width, int(x0 + patch_dim))
            patch_array = img_array[0:height, x0:x1]
            patch = (
                Image.fromarray(patch_array.squeeze())
                if patch_array.shape[2] == 1
                else Image.fromarray(patch_array)
            )

            all_patches.append(patch)
            all_coords.append((x0, 0, x1, height))
    elif direction == 1:  # path vertically
        patch_dim = height / (
            1 + (num_patches - 1) * (1 - overlap)
        )  # The patch dimension
        step = patch_dim * (
            1 - overlap
        )  # The step, i.e. how much to move to initiate the next patch

        for i in range(num_patches):
            x0 = max(0, int(i * step))
            x1 = min(height, int(x0 + patch_dim))
            patch_array = img_array[x0:x1, 0:width]
            patch = (
                Image.fromarray(patch_array.squeeze())
                if patch_array.shape[2] == 1
                else Image.fromarray(patch_array)
            )

            all_patches.append(patch)
            all_coords.append((0, x0, width, x1))

    return all_patches, all_coords


def qwen_2_5_image_scaler(
    img: Image.Image,
    target_size: int = 1024,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
) -> Tuple[Image.Image, int, int]:
    w, h = img.size

    max_dim = max([w, h])
    ratio = target_size / max_dim
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized_w = int(floor(new_w / 28) * 28)
    resized_h = int(floor(new_h / 28) * 28)

    pixels = resized_w * resized_h

    if pixels > max_pixels:
        p_ratio = sqrt(max_pixels / pixels)

        resized_w = resized_w * p_ratio
        resized_h = resized_h * p_ratio

        resized_w = int(floor(resized_w / 28) * 28)
        resized_h = int(floor(resized_h / 28) * 28)

    if pixels < min_pixels:
        p_ratio = sqrt(min_pixels / pixels)

        resized_w = resized_w * p_ratio
        resized_h = resized_h * p_ratio

        resized_w = int(ceil(resized_w / 28) * 28)
        resized_h = int(ceil(resized_h / 28) * 28)

    img = img.resize((resized_w, resized_h))

    return img, resized_w, resized_h


def internvl_2_5_image_scaler(
    img: Image.Image,
    target_size: int = 1024,
) -> Tuple[Image.Image, int, int]:
    w, h = img.size

    max_dim = max([w, h])
    ratio = target_size / max_dim
    new_w, new_h = int(w * ratio), int(h * ratio)

    img = img.resize((new_w, new_h))

    return img, new_w, new_h
