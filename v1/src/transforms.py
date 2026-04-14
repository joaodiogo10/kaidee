"""Shared image transforms used by compile, live, and render."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RectBivariateSpline


def make_abstract(img: np.ndarray, sigma: float = 50) -> np.ndarray:
    """Heavy blur + saturation boost. Destroys structure, keeps color essence."""
    f = img.astype(np.float32)
    blurred = np.stack([gaussian_filter(f[:, :, c], sigma=sigma) for c in range(3)], axis=2)
    mean = np.mean(blurred, axis=2, keepdims=True)
    blurred = mean + (blurred - mean) * 2.0
    return np.clip(blurred, 0, 255).astype(np.uint8)


def solarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Fold brightness: pixels above threshold get inverted."""
    f = img.astype(np.float32)
    mask = f > threshold
    f[mask] = 255.0 - f[mask]
    return np.clip(f, 0, 255).astype(np.uint8)


def color_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate colors by mixing RGB channels via rotation around (1,1,1) axis."""
    f = img.astype(np.float32)
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    avg = (r + g + b) / 3.0
    dr, dg, db = r - avg, g - avg, b - avg
    nr = avg + cos_a * dr + sin_a * (dg - db) * 0.577
    ng = avg + cos_a * dg + sin_a * (db - dr) * 0.577
    nb = avg + cos_a * db + sin_a * (dr - dg) * 0.577
    return np.clip(np.stack([nr, ng, nb], axis=2), 0, 255).astype(np.uint8)


def color_rotate_f32(frame: np.ndarray, angle_rad: float) -> np.ndarray:
    """Color rotate on float32 frame (in-place friendly, returns float32)."""
    cos_a = np.float32(np.cos(angle_rad))
    sin_a = np.float32(np.sin(angle_rad))
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    avg = (r + g + b) / np.float32(3.0)
    dr, dg, db = r - avg, g - avg, b - avg
    k = np.float32(0.577)
    new_r = avg + cos_a * dr + sin_a * (dg - db) * k
    new_g = avg + cos_a * dg + sin_a * (db - dr) * k
    new_b = avg + cos_a * db + sin_a * (dr - dg) * k
    return np.stack([new_r, new_g, new_b], axis=2)


def posterize(img: np.ndarray, levels: int) -> np.ndarray:
    f = img.astype(np.float32)
    step = 255.0 / (levels - 1)
    return np.clip(np.round(f / step) * step, 0, 255).astype(np.uint8)


def pixel_sort(img: np.ndarray) -> np.ndarray:
    """Sort each row's pixels by brightness."""
    f = img.astype(np.float32)
    luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    idx = np.argsort(luma, axis=1)
    result = np.take_along_axis(f, idx[:, :, np.newaxis], axis=1)
    return np.clip(result, 0, 255).astype(np.uint8)


def radial_displace(img: np.ndarray, amplitude: float = 30,
                    freq: float = 3, phase: float = 0) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    dy, dx = yy - cy, xx - cx
    radius = np.sqrt(dy ** 2 + dx ** 2)
    safe_r = np.maximum(radius, 0.001)
    disp = amplitude * np.sin(radius * freq * 2 * np.pi / max(h, w) - phase)
    new_yy = np.clip(yy + disp * dy / safe_r, 0, h - 1)
    new_xx = np.clip(xx + disp * dx / safe_r, 0, w - 1)
    result = np.zeros_like(f)
    for c in range(3):
        result[:, :, c] = map_coordinates(f[:, :, c], [new_yy, new_xx],
                                          order=1, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def fluid_displace(img: np.ndarray, amplitude: float = 40,
                   turbulence: float = 2.0, phase: float = 0) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    grid_h, grid_w = 16, 16
    gy = np.linspace(0, 1, grid_h)
    gx = np.linspace(0, 1, grid_w)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    disp_y = amplitude * np.sin(GY * turbulence * 2 * np.pi + phase) * \
        np.cos(GX * turbulence * 1.5 * np.pi + phase * 0.7)
    disp_x = amplitude * np.cos(GY * turbulence * 1.7 * np.pi + phase * 0.8) * \
        np.sin(GX * turbulence * 2 * np.pi + phase)
    full_gy, full_gx = np.linspace(0, 1, h), np.linspace(0, 1, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    new_yy = np.clip(yy + RectBivariateSpline(gy, gx, disp_y, kx=3, ky=3)(full_gy, full_gx), 0, h - 1)
    new_xx = np.clip(xx + RectBivariateSpline(gy, gx, disp_x, kx=3, ky=3)(full_gy, full_gx), 0, w - 1)
    result = np.zeros_like(f)
    for c in range(3):
        result[:, :, c] = map_coordinates(f[:, :, c], [new_yy, new_xx],
                                          order=1, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def multi_pass_abstract(img: np.ndarray, passes: int = 3) -> np.ndarray:
    """Multiple blur + solarize + color rotate passes for deep abstraction."""
    f = img.astype(np.float32)
    for i in range(passes):
        for c in range(3):
            f[:, :, c] = gaussian_filter(f[:, :, c], sigma=8 + i * 4)
        mask = f > (100 + i * 30)
        f[mask] = 255.0 - f[mask]
        angle = (i + 1) * 1.2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        avg = (r + g + b) / 3.0
        dr, dg, db = r - avg, g - avg, b - avg
        f[:, :, 0] = avg + cos_a * dr + sin_a * (dg - db) * 0.577
        f[:, :, 1] = avg + cos_a * dg + sin_a * (db - dr) * 0.577
        f[:, :, 2] = avg + cos_a * db + sin_a * (dr - dg) * 0.577
    return np.clip(f, 0, 255).astype(np.uint8)


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Fast alpha blend: a*(1-alpha) + b*alpha via uint16 integer ops."""
    ai = max(0, min(256, int(alpha * 256)))
    if ai == 0:
        return a
    if ai >= 256:
        return b
    inv = 256 - ai
    out = a.astype(np.uint16)
    out *= inv
    tmp = b.astype(np.uint16)
    tmp *= ai
    out += tmp
    out >>= 8
    return out.astype(np.uint8)


def fast_zoom(frame: np.ndarray, factor: float) -> np.ndarray:
    """Zoom by factor: >1 crops center (zoom in), <1 shrinks + black border (zoom out)."""
    h, w = frame.shape[:2]
    if factor >= 1.0:
        ch, cw = max(2, int(h / factor)), max(2, int(w / factor))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        cropped = frame[y0:y0 + ch, x0:x0 + cw]
        y_idx = np.clip((np.arange(h) * ch // h), 0, ch - 1)
        x_idx = np.clip((np.arange(w) * cw // w), 0, cw - 1)
        return cropped[y_idx][:, x_idx]
    else:
        nh, nw = max(2, int(h * factor)), max(2, int(w * factor))
        y_idx = np.clip((np.arange(nh) * h // nh), 0, h - 1)
        x_idx = np.clip((np.arange(nw) * w // nw), 0, w - 1)
        shrunk = frame[y_idx][:, x_idx]
        out = np.zeros_like(frame)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = shrunk
        return out


def make_vignette(h: int, w: int) -> np.ndarray:
    """Generate a vignette mask (float32, 0.3-1.0 range)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1.0 - 0.5 * ((yy / h - 0.5) ** 2 + (xx / w - 0.5) ** 2) * 4
    return np.clip(vig, 0.3, 1.0).astype(np.float32)
