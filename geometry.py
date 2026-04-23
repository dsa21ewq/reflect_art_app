from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SceneParams:
    paper_width_mm: float = 210.0
    paper_height_mm: float = 297.0
    cylinder_radius_mm: float = 20.0
    cylinder_height_mm: float = 120.0
    cylinder_center_x_mm: float = 105.0
    cylinder_center_y_mm: float = 150.0
    eye_x_mm: float = 105.0
    eye_y_mm: float = -180.0
    eye_z_mm: float = 320.0
    theta_samples: int = 300
    height_samples: int = 300
    theta_span_deg: float = 120.0

    @property
    def theta_center(self) -> float:
        return float(np.arctan2(self.eye_y_mm - self.cylinder_center_y_mm, self.eye_x_mm - self.cylinder_center_x_mm))

    @property
    def theta_half_span_rad(self) -> float:
        return float(np.deg2rad(self.theta_span_deg / 2.0))

    @property
    def eye(self) -> np.ndarray:
        return np.array([self.eye_x_mm, self.eye_y_mm, self.eye_z_mm], dtype=np.float64)


def make_parameter_grid(params: SceneParams) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(
        params.theta_center - params.theta_half_span_rad,
        params.theta_center + params.theta_half_span_rad,
        params.theta_samples,
        dtype=np.float64,
    )
    height = np.linspace(0.0, params.cylinder_height_mm, params.height_samples, dtype=np.float64)
    return np.meshgrid(theta, height, indexing="xy")


def compute_mapping(params: SceneParams) -> Dict[str, np.ndarray]:
    theta_grid, h_grid = make_parameter_grid(params)
    cos_t = np.cos(theta_grid)
    sin_t = np.sin(theta_grid)

    mx = params.cylinder_center_x_mm + params.cylinder_radius_mm * cos_t
    my = params.cylinder_center_y_mm + params.cylinder_radius_mm * sin_t
    mz = h_grid

    n = np.stack([cos_t, sin_t, np.zeros_like(cos_t)], axis=-1)
    m = np.stack([mx, my, mz], axis=-1)

    eye = params.eye.reshape(1, 1, 3)
    r = m - eye
    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    r_norm = np.maximum(r_norm, 1e-9)
    r_unit = r / r_norm

    dot_rn = np.sum(r_unit * n, axis=-1, keepdims=True)
    q = r_unit - 2.0 * dot_rn * n

    qz = q[..., 2]
    valid_down = qz < -1e-9

    lam = np.full_like(qz, np.nan, dtype=np.float64)
    lam[valid_down] = -mz[valid_down] / qz[valid_down]

    p = m + lam[..., None] * q
    px = p[..., 0]
    py = p[..., 1]

    inside_a4 = (
        valid_down
        & (px >= 0.0)
        & (px <= params.paper_width_mm)
        & (py >= 0.0)
        & (py <= params.paper_height_mm)
    )

    return {
        "theta_grid": theta_grid,
        "height_grid": h_grid,
        "mirror_points": m,
        "normals": n,
        "reflected_dirs": q,
        "paper_x_mm": px,
        "paper_y_mm": py,
        "valid_down": valid_down,
        "inside_a4": inside_a4,
        "lambda": lam,
    }


def compute_distortion(mapping: Dict[str, np.ndarray]) -> np.ndarray:
    px = mapping["paper_x_mm"]
    py = mapping["paper_y_mm"]
    theta = mapping["theta_grid"]
    h = mapping["height_grid"]
    valid = mapping["inside_a4"]

    dpx_dtheta, dpx_dh = np.gradient(px, axis=(1, 0))
    dpy_dtheta, dpy_dh = np.gradient(py, axis=(1, 0))
    dtheta = np.gradient(theta, axis=1)
    dh = np.gradient(h, axis=0)

    jac = (dpx_dtheta * dpy_dh - dpx_dh * dpy_dtheta) / np.maximum(dtheta * dh, 1e-9)
    jac_abs = np.abs(jac)
    jac_abs[~valid] = np.nan
    return jac_abs


def summarize_mapping(mapping: Dict[str, np.ndarray], params: SceneParams) -> Dict[str, float]:
    valid_down = mapping["valid_down"]
    inside = mapping["inside_a4"]
    total = float(valid_down.size)
    valid_count = float(np.count_nonzero(valid_down))
    inside_count = float(np.count_nonzero(inside))

    px = mapping["paper_x_mm"]
    py = mapping["paper_y_mm"]
    finite_mask = np.isfinite(px) & np.isfinite(py)
    bbox_mask = valid_down & finite_mask

    if np.any(bbox_mask):
        xmin = float(np.nanmin(px[bbox_mask]))
        xmax = float(np.nanmax(px[bbox_mask]))
        ymin = float(np.nanmin(py[bbox_mask]))
        ymax = float(np.nanmax(py[bbox_mask]))
    else:
        xmin = xmax = ymin = ymax = np.nan

    return {
        "valid_reflection_ratio": valid_count / total,
        "inside_a4_ratio": inside_count / total,
        "inside_given_valid_ratio": inside_count / max(valid_count, 1.0),
        "paper_bbox_xmin_mm": xmin,
        "paper_bbox_xmax_mm": xmax,
        "paper_bbox_ymin_mm": ymin,
        "paper_bbox_ymax_mm": ymax,
        "eye_above_cylinder": float(params.eye_z_mm > params.cylinder_height_mm),
    }

