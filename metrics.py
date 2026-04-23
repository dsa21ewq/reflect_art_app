from __future__ import annotations

from typing import Dict

import numpy as np


def mse(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    if mask is not None:
        mask3 = np.repeat(mask[..., None], 3, axis=-1)
        diff = a[mask3] - b[mask3]
    else:
        diff = a - b
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff ** 2))


def psnr(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = mse(image_a, image_b, mask=mask)
    if not np.isfinite(err):
        return float("nan")
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(err)))


def ssim_approx(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.reshape(-1, 3).mean(axis=1)
    b = b.reshape(-1, 3).mean(axis=1)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    sigma_a = float(np.var(a))
    sigma_b = float(np.var(b))
    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(num / max(den, 1e-9))


def distortion_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "distortion_mean": float("nan"),
            "distortion_p90": float("nan"),
            "distortion_p95": float("nan"),
            "distortion_max": float("nan"),
        }
    return {
        "distortion_mean": float(np.mean(finite)),
        "distortion_p90": float(np.percentile(finite, 90)),
        "distortion_p95": float(np.percentile(finite, 95)),
        "distortion_max": float(np.max(finite)),
    }

