"""Base class for visual modes."""

from __future__ import annotations
from typing import Protocol
import numpy as np

from src.assets import Assets
from src.params import DerivedParams
from src.drift import DriftValues


class ImageBlender(Protocol):
    """Protocol for the image blending callback provided by the renderer."""
    def __call__(self, t: float, variant: str) -> np.ndarray | None: ...


class FrameKeyFn(Protocol):
    """Protocol for building asset keys synced to BPM."""
    def __call__(self, t: float, variant: str) -> str | None: ...


class Mode:
    """Base class for visual modes. Each mode owns its own state."""

    # Whether this mode uses sharp (unblurred) images as input
    uses_sharp: bool = False

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def reset(self, w: int, h: int) -> None:
        """Reset internal state (e.g. after resolution change)."""
        self.w = w
        self.h = h

    def render(
        self,
        t: float,
        bass: float,
        mids: float,
        highs: float,
        energy: float,
        onset: float,
        dp: DerivedParams,
        drift: DriftValues,
        assets: Assets,
        img_blend: ImageBlender,
        frame_key: FrameKeyFn,
        fps: float,
    ) -> np.ndarray:
        """Render one frame. Must be overridden by subclasses."""
        raise NotImplementedError
