"""Load pre-compiled visual assets into RAM with lazy variant derivation."""

import json
import os
import re
import sys

import numpy as np

from src.transforms import solarize, color_rotate, posterize


# Regex to detect derivable variant keys and extract (prefix, operation, param)
_DERIVE_RE = re.compile(
    r'^(.+?)_(inverted|solar_(\d+)|crot_(\d+)|poster_(\d+))$'
)

# Max cached (derived + downscaled) frames in the LRU
_CACHE_MAX = 40


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
        self._base_frames: dict[str, np.ndarray] = {}
        for key in self.meta["frame_keys"]:
            path = os.path.join(directory, f"{key}.npz")
            if os.path.exists(path):
                self._base_frames[key] = np.load(path)['data']
            else:
                npy_path = os.path.join(directory, f"{key}.npy")
                if os.path.exists(npy_path):
                    self._base_frames[key] = np.load(npy_path)

        vig_path = os.path.join(directory, "vignette.npz")
        if os.path.exists(vig_path):
            self._full_vignette: np.ndarray | None = np.load(vig_path)['data']
        else:
            npy_vig = os.path.join(directory, "vignette.npy")
            self._full_vignette = np.load(npy_vig) if os.path.exists(npy_vig) else None
        self.vignette = self._full_vignette

        # Scale state (set by rescale())
        self._scale_y_idx: np.ndarray | None = None
        self._scale_x_idx: np.ndarray | None = None

        # LRU cache for derived + downscaled frames
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []

        # Count derivable variants for reporting
        n_abstract = sum(1 for k in self._base_frames if k.endswith('_abstract'))
        n_derivable = n_abstract * 12  # inverted + 3 solar + 5 crot + 3 poster
        print(f"  {len(self._base_frames)} base frames loaded "
              f"({n_derivable} variants derived on demand)")

    def _try_derive(self, key: str) -> np.ndarray | None:
        """Attempt to compute a derived variant from an abstract base."""
        m = _DERIVE_RE.match(key)
        if not m:
            return None
        prefix = m.group(1)
        abstract_key = f"{prefix}_abstract"
        abstract = self._base_frames.get(abstract_key)
        if abstract is None:
            return None

        variant = m.group(2)
        if variant == "inverted":
            return (255 - abstract).astype(np.uint8)
        elif variant.startswith("solar_"):
            return solarize(abstract, int(m.group(3)))
        elif variant.startswith("crot_"):
            return color_rotate(abstract, int(m.group(4)))
        elif variant.startswith("poster_"):
            return posterize(abstract, int(m.group(5)))
        return None

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame using precomputed indices (nearest neighbor)."""
        if self._scale_y_idx is not None:
            return frame[self._scale_y_idx][:, self._scale_x_idx]
        return frame

    def _cache_put(self, key: str, frame: np.ndarray) -> None:
        """Add frame to LRU cache, evicting oldest if full."""
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = frame
        self._cache_order.append(key)
        while len(self._cache) > _CACHE_MAX:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)

    def get(self, key: str) -> np.ndarray | None:
        # Fast path: LRU cache hit
        cached = self._cache.get(key)
        if cached is not None:
            # Move to end of LRU
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return cached

        # Try base frames first
        frame = self._base_frames.get(key)
        if frame is None:
            # Try lazy derivation
            frame = self._try_derive(key)
        if frame is None:
            return None

        # Downscale if render scale < 1.0
        result = self._downscale(frame)
        self._cache_put(key, result)
        return result

    def rescale(self, full_w: int, full_h: int, scale: float) -> None:
        """Set render scale. Clears the cache; get() downscales on demand."""
        w = int(full_w * scale)
        h = int(full_h * scale)

        if scale < 1.0:
            self._scale_y_idx = np.linspace(0, full_h - 1, h).astype(int)
            self._scale_x_idx = np.linspace(0, full_w - 1, w).astype(int)
        else:
            self._scale_y_idx = None
            self._scale_x_idx = None

        # Downscale vignette
        if self._full_vignette is not None:
            if scale < 1.0:
                self.vignette = self._full_vignette[self._scale_y_idx][:, self._scale_x_idx]
            else:
                self.vignette = self._full_vignette

        # Invalidate cache — scale changed
        self._cache.clear()
        self._cache_order.clear()
