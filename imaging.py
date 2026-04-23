from __future__ import annotations

from io import BytesIO
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from geometry import SceneParams


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ensure_rgb(image).resize(size, resample=Image.Resampling.LANCZOS)


def compose_image(
    image: Image.Image,
    canvas_size: Tuple[int, int],
    mode: str = "contain",
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image = ensure_rgb(image)
    cw, ch = canvas_size
    iw, ih = image.size
    if mode == "stretch":
        return image.resize(canvas_size, resample=Image.Resampling.LANCZOS)
    if mode == "contain":
        scale = min(cw / max(iw, 1), ch / max(ih, 1))
    elif mode == "cover":
        scale = max(cw / max(iw, 1), ch / max(ih, 1))
    else:
        raise ValueError(f"Unknown compose mode: {mode}")
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = image.resize((nw, nh), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, bg_color)
    left = (cw - nw) // 2
    top = (ch - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_image_to_mask(
    image: Image.Image,
    canvas_size: Tuple[int, int],
    mask: np.ndarray | None = None,
    mode: str = "contain",
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    pad_ratio: float = 0.02,
) -> Image.Image:
    if mask is None or not np.any(mask):
        return compose_image(image, canvas_size, mode=mode, bg_color=bg_color)
    bbox = mask_bbox(mask)
    if bbox is None:
        return compose_image(image, canvas_size, mode=mode, bg_color=bg_color)
    cw, ch = canvas_size
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))
    x0 = max(0, x0 + pad_x)
    y0 = max(0, y0 + pad_y)
    x1 = min(cw - 1, x1 - pad_x)
    y1 = min(ch - 1, y1 - pad_y)
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    placed = compose_image(image, (box_w, box_h), mode=mode, bg_color=bg_color)
    canvas = Image.new("RGB", canvas_size, bg_color)
    canvas.paste(placed, (x0, y0))
    return canvas


def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def default_paper_image(size: Tuple[int, int] = (700, 990)) -> Image.Image:
    w, h = size
    image = Image.new("RGB", size, (248, 246, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, w - 30, h - 30), outline=(40, 40, 40), width=4)
    for t in np.linspace(0.15, 0.85, 6):
        x = int(w * t)
        draw.line((x, 60, x, h - 60), fill=(215, 215, 215), width=2)
    for t in np.linspace(0.12, 0.88, 7):
        y = int(h * t)
        draw.line((60, y, w - 60, y), fill=(220, 220, 220), width=2)
    try:
        font_big = ImageFont.truetype("Arial Unicode.ttf", 92)
        font_small = ImageFont.truetype("Arial Unicode.ttf", 42)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((90, 90), "A4 PAPER", fill=(25, 25, 25), font=font_big)
    draw.text((100, 200), "Reflect Art", fill=(160, 40, 40), font=font_small)
    draw.ellipse((180, 310, 520, 650), outline=(30, 90, 170), width=8)
    draw.line((200, 620, 500, 350), fill=(210, 100, 30), width=10)
    draw.line((180, 720, 540, 720), fill=(60, 60, 60), width=6)
    draw.text((170, 760), "forward simulation demo", fill=(60, 60, 60), font=font_small)
    return image


def default_target_image(size: Tuple[int, int] = (480, 480)) -> Image.Image:
    w, h = size
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 80, 410, 420), fill=(247, 180, 95), outline=(120, 70, 30), width=6)
    draw.polygon([(125, 160), (165, 110), (205, 165)], fill=(248, 200, 150), outline=(120, 70, 30))
    draw.polygon([(275, 165), (315, 110), (355, 160)], fill=(248, 200, 150), outline=(120, 70, 30))
    draw.ellipse((150, 210, 220, 280), fill=(255, 255, 255), outline=(90, 60, 40), width=4)
    draw.ellipse((260, 210, 330, 280), fill=(255, 255, 255), outline=(90, 60, 40), width=4)
    draw.ellipse((178, 233, 198, 253), fill=(70, 40, 20))
    draw.ellipse((288, 233, 308, 253), fill=(70, 40, 20))
    draw.polygon([(240, 275), (262, 302), (218, 302)], fill=(240, 120, 120), outline=(120, 60, 60))
    draw.arc((185, 285, 295, 350), start=15, end=165, fill=(100, 40, 30), width=4)
    for y in (286, 300, 316):
        draw.line((148, y, 95, y - 6), fill=(110, 70, 40), width=3)
        draw.line((332, y, 385, y - 6), fill=(110, 70, 40), width=3)
    draw.text((145, 26), "CAT", fill=(20, 20, 20))
    return image


def default_gray_portrait(size: Tuple[int, int] = (420, 560)) -> Image.Image:
    w, h = size
    img = Image.new("L", size, 225)
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 60, 340, 350), fill=180)
    draw.ellipse((130, 120, 180, 180), fill=65)
    draw.ellipse((240, 120, 290, 180), fill=65)
    draw.ellipse((145, 135, 165, 155), fill=235)
    draw.ellipse((255, 135, 275, 155), fill=235)
    draw.rectangle((195, 150, 225, 245), fill=120)
    draw.arc((150, 200, 270, 285), start=20, end=160, fill=60, width=7)
    draw.rectangle((110, 335, 315, 530), fill=105)
    draw.ellipse((50, 25, 370, 380), outline=95, width=8)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.1))
    return img.convert("RGB")


def sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray, fill_value: float = 255.0) -> np.ndarray:
    h, w, c = image.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full(x.shape + (c,), fill_value, dtype=np.float32)
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    dx = (xv - x0v).reshape(-1, 1)
    dy = (yv - y0v).reshape(-1, 1)

    c00 = image[y0v, x0v]
    c10 = image[y0v, x1v]
    c01 = image[y1v, x0v]
    c11 = image[y1v, x1v]

    interp = (
        c00 * (1.0 - dx) * (1.0 - dy)
        + c10 * dx * (1.0 - dy)
        + c01 * (1.0 - dx) * dy
        + c11 * dx * dy
    )
    out[valid] = interp
    return out


def mm_to_pixel_coords(params: SceneParams, x_mm: np.ndarray, y_mm: np.ndarray, width_px: int, height_px: int) -> Tuple[np.ndarray, np.ndarray]:
    px = x_mm / params.paper_width_mm * (width_px - 1)
    py = y_mm / params.paper_height_mm * (height_px - 1)
    return px, py


def tint_invalid_regions(image: np.ndarray, valid_mask: np.ndarray, tint_rgb: Tuple[float, float, float] = (255.0, 210.0, 210.0), alpha: float = 0.65) -> np.ndarray:
    out = image.copy()
    invalid = ~valid_mask
    if np.any(invalid):
        tint = np.array(tint_rgb, dtype=np.float32)
        out[invalid] = alpha * out[invalid] + (1.0 - alpha) * tint
    return out


def forward_simulation(paper_image: Image.Image, mapping: Dict[str, np.ndarray], params: SceneParams, out_width: int, out_height: int) -> Dict[str, np.ndarray]:
    paper_arr = pil_to_array(resize_image(paper_image, (out_width, out_height)))
    px = mapping["paper_x_mm"]
    py = mapping["paper_y_mm"]
    inside = mapping["inside_a4"]
    grid_x, grid_y = mm_to_pixel_coords(params, px, py, out_width, out_height)
    sampled = sample_bilinear(paper_arr, grid_x, grid_y, fill_value=255.0)
    sampled[~inside] = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return {
        "paper_array": paper_arr,
        "mirror_array": sampled,
        "valid_mask": inside,
    }


def inverse_design(target_mirror: Image.Image, mapping: Dict[str, np.ndarray], params: SceneParams, paper_width_px: int, paper_height_px: int, fill_color: int = 250) -> Dict[str, np.ndarray]:
    target_arr = pil_to_array(resize_image(target_mirror, (params.theta_samples, params.height_samples)))
    canvas = np.zeros((paper_height_px, paper_width_px, 3), dtype=np.float32)
    weight = np.zeros((paper_height_px, paper_width_px), dtype=np.float32)
    inside = mapping["inside_a4"]
    px_mm = mapping["paper_x_mm"]
    py_mm = mapping["paper_y_mm"]
    px, py = mm_to_pixel_coords(params, px_mm, py_mm, paper_width_px, paper_height_px)

    for dy in (0, 1):
        for dx in (0, 1):
            wx = (1.0 - np.abs(px - (np.floor(px) + dx)))
            wy = (1.0 - np.abs(py - (np.floor(py) + dy)))
            w = np.clip(wx, 0.0, 1.0) * np.clip(wy, 0.0, 1.0) * inside

            xi = np.clip((np.floor(px).astype(np.int32) + dx), 0, paper_width_px - 1)
            yi = np.clip((np.floor(py).astype(np.int32) + dy), 0, paper_height_px - 1)

            np.add.at(weight, (yi.ravel(), xi.ravel()), w.ravel())
            for ch in range(3):
                np.add.at(canvas[..., ch], (yi.ravel(), xi.ravel()), (target_arr[..., ch] * w).ravel())

    paper = np.full_like(canvas, fill_color, dtype=np.float32)
    valid_weight = weight > 1e-6
    paper[valid_weight] = canvas[valid_weight] / weight[valid_weight, None]
    paper = hole_fill_image(paper, valid_weight, passes=2)
    return {
        "target_array": target_arr,
        "paper_array": paper,
        "paper_weight": weight,
        "paper_mask": valid_weight,
    }


def paper_usage_bbox(weight: np.ndarray) -> Tuple[int, int, int, int] | None:
    return mask_bbox(weight > 1e-6)


def draw_bbox_overlay(image: Image.Image, bbox: Tuple[int, int, int, int] | None, color: Tuple[int, int, int] = (255, 80, 80), width: int = 3) -> Image.Image:
    out = image.copy()
    if bbox is None:
        return out
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=color, width=width)
    return out


def hole_fill_image(image: np.ndarray, mask: np.ndarray, passes: int = 1) -> np.ndarray:
    out = image.copy()
    filled = mask.copy()
    for _ in range(passes):
        if np.all(filled):
            break
        new_out = out.copy()
        for shift_y, shift_x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rolled = np.roll(np.roll(out, shift_y, axis=0), shift_x, axis=1)
            rolled_mask = np.roll(np.roll(filled, shift_y, axis=0), shift_x, axis=1)
            add_mask = (~filled) & rolled_mask
            new_out[add_mask] += rolled[add_mask]
        count = np.zeros(mask.shape, dtype=np.float32)
        for shift_y, shift_x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rolled_mask = np.roll(np.roll(filled, shift_y, axis=0), shift_x, axis=1)
            count += rolled_mask.astype(np.float32)
        update = (~filled) & (count > 0)
        new_out[update] /= count[update, None]
        out[update] = new_out[update]
        filled[update] = True
    return out


def make_heatmap(values: np.ndarray, invalid_mask: np.ndarray | None = None) -> Image.Image:
    import matplotlib.cm as cm

    arr = values.astype(np.float32)
    finite = np.isfinite(arr)
    if invalid_mask is not None:
        finite &= ~invalid_mask
    if np.any(finite):
        vmin = np.nanpercentile(arr[finite], 5)
        vmax = np.nanpercentile(arr[finite], 95)
        scale = (arr - vmin) / max(vmax - vmin, 1e-6)
    else:
        scale = np.zeros_like(arr)
    scale = np.clip(scale, 0.0, 1.0)
    rgba = cm.turbo(scale)
    rgba[..., 3] = np.where(finite, 1.0, 0.0)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_grid(image: Image.Image, step_px: int = 50, color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for x in range(step_px, w, step_px):
        draw.line((x, 0, x, h), fill=color + (128,), width=1)
    for y in range(step_px, h, step_px):
        draw.line((0, y, w, y), fill=color + (128,), width=1)
    return out


def build_default_input(kind: str) -> Image.Image:
    if kind == "paper":
        return default_paper_image()
    if kind == "mirror":
        return default_target_image()
    if kind == "portrait":
        return default_gray_portrait()
    return default_target_image()
