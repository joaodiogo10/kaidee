"""Master parameters and derived values."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Params:
    """Master controls — all 0.0 to 1.0."""
    reactivity: float = 0.5
    perception: float = 0.5
    movement: float = 0.5
    color: float = 0.5
    brightness: float = 0.5
    blend: float = 0.5
    img_blend: float = 0.5
    code: float = 0.5
    zoom: float = 0.0
    pan_x: float = 0.5
    pan_y: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def adjust(self, name: str, delta: float) -> None:
        val = getattr(self, name)
        setattr(self, name, round(max(0.0, min(1.0, val + delta)), 2))


@dataclass
class DerivedParams:
    """Values computed from master Params — used by modes and renderer."""
    # From reactivity
    react_mod: float = 1.0
    code_vis: float = 0.5
    code_blend: float = 0.5
    bass_gain: float = 1.2
    mids_gain: float = 1.0
    highs_gain: float = 1.0

    # From perception
    warp: float = 0.5
    overlap: float = 0.4
    trail: float = 0.3
    chaos_manual: float = -1.0
    img_cycle_bars: int = 4

    # From movement
    rotation_speed: float = 0.5
    zoom_amount: float = 0.5
    drift_speed: float = 0.5
    pan_amount: float = 0.5

    # From color
    chroma: float = 0.5
    color_intensity: float = 1.0
    color_shift_speed: float = 1.0

    # From img_blend
    img_crossfade: float = 0.5
    img_layers: int = 1
    img_layer_alpha: float = 0.0


def derive(p: Params) -> DerivedParams:
    """Compute all derived values from master controls."""
    r = p.reactivity ** 2
    perc = p.perception ** 2.5
    m = p.movement ** 2.5
    c = p.color ** 2.5
    ib = p.img_blend

    return DerivedParams(
        # Reactivity
        react_mod=0.2 + r * 2.8,
        code_vis=p.code * 2.0,
        code_blend=p.code,
        bass_gain=1.2,
        mids_gain=1.0,
        highs_gain=1.0,
        # Perception
        warp=perc,
        overlap=perc * 0.8,
        trail=perc * 0.6,
        chaos_manual=perc * 0.8 if perc > 0.1 else -1.0,
        img_cycle_bars=[16, 8, 4, 2][min(3, int(p.perception * 4))],
        # Movement
        rotation_speed=m * 12.0,
        zoom_amount=m * 2.5,
        drift_speed=p.movement ** 1.5,
        pan_amount=p.movement ** 1.5,
        # Color
        chroma=c * 1.5,
        color_intensity=0.3 + c * 2.5,
        color_shift_speed=c * 4.0,
        # Image blend
        img_crossfade=min(1.0, ib * 2),
        img_layers=max(1, int(1 + ib * 5)),
        img_layer_alpha=ib * 0.7,
    )


PRESET_FIELDS = ["reactivity", "perception", "movement", "color",
                 "brightness", "blend", "img_blend", "zoom", "pan_x", "pan_y"]


@dataclass
class Preset:
    params: Params
    mode: str = "auto"

    def to_dict(self) -> dict:
        d = self.params.to_dict()
        d["mode"] = self.mode
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Preset:
        mode = d.get("mode", "auto")
        params = Params.from_dict(d)
        return cls(params=params, mode=mode)


def load_presets(path: str) -> dict[str, Preset]:
    try:
        with open(path) as f:
            raw = json.load(f)
        return {k: Preset.from_dict(v) for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}


def save_presets(presets: dict[str, Preset], path: str) -> None:
    raw = {k: v.to_dict() for k, v in presets.items()}
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
