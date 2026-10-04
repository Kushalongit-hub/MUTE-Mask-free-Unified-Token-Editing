"""Frozen HSV color-surface proxies; these are not full object annotations."""
from collections import deque
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def color_mask(image, color, hsv_config):
    hsv = np.asarray(image.convert("HSV"), dtype=float) / 255
    hue = np.zeros(hsv.shape[:2], dtype=bool)
    for low, high in hsv_config[color]:
        hue |= (hsv[..., 0] >= low) & (hsv[..., 0] <= high)
    return hue & (hsv[..., 1] >= hsv_config["minimum_saturation"]) & (hsv[..., 2] >= hsv_config["minimum_value"])


def components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask)
    for y, x in zip(*np.where(mask)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(int(y), int(x))])
        pixels = []
        while queue:
            cy, cx = queue.popleft()
            pixels.append(cy * width + cx)
            for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                if 0 <= ny < height and 0 <= nx < width and mask[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx] = True
                    queue.append((ny,nx))
        yield np.asarray(pixels, dtype=int)


def source_mask(image, protocol):
    colored = color_mask(image, "red", protocol["hsv"])
    largest = max(components(colored), key=len, default=np.array([], dtype=int))
    result = np.zeros_like(colored)
    result.flat[largest] = True
    return result


def dilate(mask, radius):
    tensor = torch.from_numpy(mask.astype(np.float32))[None,None]
    return F.max_pool2d(tensor, 2*radius+1, stride=1, padding=radius)[0,0].numpy().astype(bool)


def score(source, edited, frozen_mask, color, protocol):
    if source.size != edited.size or frozen_mask.shape != (source.height, source.width):
        raise ValueError("Evaluation dimensions differ")
    count = int(frozen_mask.sum())
    valid = protocol["source_validity"]
    source_valid = valid["minimum_pixels"] <= count <= valid["maximum_area_fraction"] * frozen_mask.size
    if not source_valid:
        return {"source_valid": False, "source_mask_pixels": count, "joint_proxy_pass": False,
                "warning": "Invalid source-colored-region mask; retain case in denominator"}
    colored = color_mask(edited, color, protocol["hsv"])
    best_pixels = np.array([], dtype=int)
    best_overlap = 0
    for pixels in components(colored):
        overlap = int(frozen_mask.flat[pixels].sum())
        if overlap > best_overlap:
            best_overlap, best_pixels = overlap, pixels
    target_mask = np.zeros_like(frozen_mask)
    target_mask.flat[best_pixels] = True
    intersection = int((frozen_mask & target_mask).sum())
    union = int((frozen_mask | target_mask).sum())
    coverage = float(colored[frozen_mask].mean())
    iou = intersection / union
    source_center = np.argwhere(frozen_mask).mean(axis=0)
    shift = float(np.linalg.norm(np.argwhere(target_mask).mean(axis=0)-source_center)) if target_mask.any() else None
    background = ~dilate(frozen_mask, 8)
    delta = np.abs(np.asarray(source, dtype=float)-np.asarray(edited, dtype=float))/255
    mae = float(delta[background].mean()) if background.any() else None
    thresholds = protocol["proxy_thresholds"]
    passed = coverage >= thresholds["color_coverage_min"] and iou >= thresholds["colored_region_iou_min"] and (
        shift is not None and shift <= thresholds["centroid_shift_max_px"]) and (
        mae is not None and mae <= thresholds["background_mae_max"])
    return {"source_valid": True, "source_mask_pixels": count, "target_color_coverage": coverage,
            "colored_region_iou": iou, "colored_region_centroid_shift_px": shift,
            "colored_region_area_ratio": float(target_mask.sum()/count),
            "independent_color_mask_background_rgb_mae": mae, "joint_proxy_pass": bool(passed),
            "warning": "HSV colored-region proxies; not independent semantic success or full-object geometry"}
