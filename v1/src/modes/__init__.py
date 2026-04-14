"""Visual mode implementations."""

from src.modes.base import Mode
from src.modes.feedback import FeedbackMode
from src.modes.glitch import GlitchMode
from src.modes.radial import RadialMode
from src.modes.kaleidoscope import KaleidoscopeMode
from src.modes.strobe import StrobeMode
from src.modes.displacement import DisplacementMode

MODE_CLASSES: dict[str, type[Mode]] = {
    "feedback": FeedbackMode,
    "glitch": GlitchMode,
    "radial": RadialMode,
    "kaleidoscope": KaleidoscopeMode,
    "strobe": StrobeMode,
    "displacement": DisplacementMode,
}

MODE_NAMES = ["feedback", "glitch", "radial", "kaleidoscope", "strobe", "displacement", "auto"]

COMPLEMENTS = {
    "feedback": "kaleidoscope",
    "glitch": "feedback",
    "radial": "kaleidoscope",
    "kaleidoscope": "displacement",
    "strobe": "glitch",
    "displacement": "radial",
}

# Auto-mode pools by energy level
_AUTO_POOLS = {
    "calm": ["feedback", "radial", "displacement"],
    "mid": ["feedback", "radial", "kaleidoscope", "displacement"],
    "intense": ["glitch", "strobe", "kaleidoscope", "feedback"],
}


def get_auto_mode_pool(mode_energy: float) -> list[str]:
    """Return the mode pool for a given energy level."""
    if mode_energy < 0.35:
        return _AUTO_POOLS["calm"]
    elif mode_energy < 0.65:
        return _AUTO_POOLS["mid"]
    return _AUTO_POOLS["intense"]


def get_auto_mode(mode_energy: float, bar_idx: int) -> str:
    """Select the active mode name for auto-mode given energy and bar index."""
    pool = get_auto_mode_pool(mode_energy)
    section = bar_idx // 4
    return pool[section % len(pool)]
