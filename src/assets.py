"""Load pre-compiled visual assets into RAM."""

import json
import os
import sys

import numpy as np

from src.transforms import solarize, color_rotate, posterize


class Assets:
    def __init__(self, directory: str):
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.exists(meta_path):
            print(f"ERROR: No compiled assets found at '{directory}/'")
            print("  Run compile.py first!")
            sys.exit(1)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.n_images: int = self.meta["n_images"]
        self.code_features: dict = self.meta["code_features"]
        self.code_images: list[str] = self.meta.get("code_images", [])
        self.code_variants: list[str] = self.meta.get(
            "code_variants",
            ["sharp", "abstract", "sorted", "deep"],
        )
        self.resolution: tuple[int, int] = tuple(self.meta["resolution"])

        print(f"Loading {len(self.meta['frame_keys'])} compiled frames...")
        self.frames: dict[str, np.ndarray] = {}
        for key in self.meta["frame_keys"]:
            path = os.path.join(directory, f"{key}.npz")
            if os.path.exists(path):
                self.frames[key] = np.load(path)['data']
            else:
                # Fallback for old .npy format
                npy_path = os.path.join(directory, f"{key}.npy")
                if os.path.exists(npy_path):
                    self.frames[key] = np.load(npy_path)

        vig_path = os.path.join(directory, "vignette.npz")
        if os.path.exists(vig_path):
            self.vignette: np.ndarray | None = np.load(vig_path)['data']
        else:
            # Fallback for old .npy format
            npy_vig = os.path.join(directory, "vignette.npy")
            self.vignette = np.load(npy_vig) if os.path.exists(npy_vig) else None

        # Derive cheap variants from abstract bases
        print("  Deriving color/effect variants...")
        self._derive_variants()

        # Cache for downscaled frames
        self._full_frames: dict[str, np.ndarray] | None = None
        self._full_vignette: np.ndarray | None = None

        print(f"  {len(self.frames)} frames ready")

    def _derive_variants(self) -> None:
        """Generate cheap variants (solarize, color_rotate, posterize, invert)
        from abstract bases. These are fast enough to compute at load time
        rather than storing on disk."""
        base_keys = [k for k in self.frames if k.endswith('_abstract')]
        for base_key in base_keys:
            prefix = base_key[:-len('_abstract')]
            abstract = self.frames[base_key]
            self.frames[f"{prefix}_inverted"] = (255 - abstract).astype(np.uint8)
            for thresh in (80, 140, 200):
                self.frames[f"{prefix}_solar_{thresh}"] = solarize(abstract, thresh)
            for angle in (60, 120, 180, 240, 300):
                self.frames[f"{prefix}_crot_{angle}"] = color_rotate(abstract, angle)
            for levels in (2, 3, 4):
                self.frames[f"{prefix}_poster_{levels}"] = posterize(abstract, levels)

    def get(self, key: str) -> np.ndarray | None:
        return self.frames.get(key)

    def rescale(self, full_w: int, full_h: int, scale: float) -> None:
        """Downscale or restore all asset frames to a target resolution."""
        w = int(full_w * scale)
        h = int(full_h * scale)

        if self._full_frames is None:
            self._full_frames = dict(self.frames)
            if self.vignette is not None:
                self._full_vignette = self.vignette.copy()

        if scale < 1.0:
            y_idx = np.linspace(0, full_h - 1, h).astype(int)
            x_idx = np.linspace(0, full_w - 1, w).astype(int)
            for key, full in self._full_frames.items():
                self.frames[key] = full[y_idx][:, x_idx]
            if self.vignette is not None and self._full_vignette is not None:
                self.vignette = self._full_vignette[y_idx][:, x_idx]
        else:
            self.frames = dict(self._full_frames)
            if self._full_vignette is not None:
                self.vignette = self._full_vignette
