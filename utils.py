from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from geometry import SceneParams


DEFAULTS: Dict[str, Any] = {
    "paper_width_mm": 210.0,
    "paper_height_mm": 297.0,
    "cylinder_radius_mm": 20.0,
    "cylinder_height_mm": 120.0,
    "cylinder_center_x_mm": 105.0,
    "cylinder_center_y_mm": 150.0,
    "eye_x_mm": 105.0,
    "eye_y_mm": -180.0,
    "eye_z_mm": 320.0,
    "theta_samples": 300,
    "height_samples": 300,
    "theta_span_deg": 120.0,
    "paper_resolution_px": 900,
    "paper_fit_mode": "contain",
    "mirror_fit_mode": "contain",
    "fit_target_to_valid_region": True,
    "fit_paper_to_canvas": True,
    "show_out_of_bounds": False,
    "show_grid": False,
    "show_heatmap": False,
    "limit_to_a4": True,
}


def params_from_dict(config: Dict[str, Any]) -> SceneParams:
    return SceneParams(
        paper_width_mm=float(config["paper_width_mm"]),
        paper_height_mm=float(config["paper_height_mm"]),
        cylinder_radius_mm=float(config["cylinder_radius_mm"]),
        cylinder_height_mm=float(config["cylinder_height_mm"]),
        cylinder_center_x_mm=float(config["cylinder_center_x_mm"]),
        cylinder_center_y_mm=float(config["cylinder_center_y_mm"]),
        eye_x_mm=float(config["eye_x_mm"]),
        eye_y_mm=float(config["eye_y_mm"]),
        eye_z_mm=float(config["eye_z_mm"]),
        theta_samples=int(config["theta_samples"]),
        height_samples=int(config["height_samples"]),
        theta_span_deg=float(config["theta_span_deg"]),
    )


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def open_uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def format_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"
