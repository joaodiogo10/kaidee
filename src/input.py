"""Input / event handling extracted from LiveRenderer."""

from __future__ import annotations

import random
import numpy as np
import pygame

from src.audio import FileAudio, DemoAudio
from src.params import Params, Preset, derive, save_presets
from src.modes import MODE_NAMES


def handle_events(renderer) -> None:
    """Process pygame events, delegating key presses."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            renderer.running = False
        elif event.type == pygame.KEYDOWN:
            handle_key(renderer, event)


def handle_key(r, event) -> None:
    """Handle a single keydown event against the renderer state."""
    key = event.key

    if key == pygame.K_ESCAPE:
        r.running = False
    elif key == pygame.K_f:
        r.fullscreen = not r.fullscreen
        # toggle_fullscreen uses SDL2 native fullscreen on the CURRENT display,
        # correctly handling multi-monitor setups (no context recreation needed)
        pygame.display.toggle_fullscreen()
    elif key == pygame.K_h:
        r.show_hud = not r.show_hud
    elif key == pygame.K_SPACE:
        if isinstance(r.audio, FileAudio):
            r.audio.toggle_pause()
    elif key == pygame.K_UP:
        if isinstance(r.audio, FileAudio):
            r.audio.set_volume(r.audio.get_volume() + 0.1)
    elif key == pygame.K_DOWN:
        if isinstance(r.audio, FileAudio):
            r.audio.set_volume(r.audio.get_volume() - 0.1)
    elif key == pygame.K_d:
        if not isinstance(r.audio, DemoAudio):
            r.audio.stop()
            r.audio = DemoAudio()
            print("Demo audio")
    # Master controls
    elif key == pygame.K_z:
        r.params.adjust("reactivity", -0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_x:
        r.params.adjust("reactivity", 0.05)
        r.dp = derive(r.params)
    elif getattr(event, 'unicode', '') == '<':
        r.params.adjust("code", -0.05)
        r.dp = derive(r.params)
    elif getattr(event, 'unicode', '') == '>':
        r.params.adjust("code", 0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_COMMA:
        r.params.adjust("perception", -0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_PERIOD:
        r.params.adjust("perception", 0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_a:
        r.params.adjust("movement", -0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_s:
        r.params.adjust("movement", 0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_b:
        r.params.adjust("color", -0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_n:
        r.params.adjust("color", 0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_j:
        r.params.adjust("brightness", -0.05)
    elif key == pygame.K_k:
        r.params.adjust("brightness", 0.05)
    elif key == pygame.K_c:
        r.params.adjust("blend", -0.05)
    elif key == pygame.K_v:
        r.params.adjust("blend", 0.05)
    elif key == pygame.K_u:
        r.params.adjust("img_blend", -0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_i:
        r.params.adjust("img_blend", 0.05)
        r.dp = derive(r.params)
    elif key == pygame.K_LEFTBRACKET or getattr(event, 'unicode', '') == '[':
        r.params.adjust("zoom", -0.05)
    elif key == pygame.K_RIGHTBRACKET or getattr(event, 'unicode', '') == ']':
        r.params.adjust("zoom", 0.05)
    elif key == pygame.K_t:
        r.params.adjust("pan_x", -0.05)
    elif key == pygame.K_y:
        r.params.adjust("pan_x", 0.05)
    elif key == pygame.K_o:
        r.params.adjust("pan_y", -0.05)
    elif key == pygame.K_p:
        r.params.adjust("pan_y", 0.05)
    elif key == pygame.K_MINUS or getattr(event, 'unicode', '') == '-':
        r.fps = max(10, r.fps - 5)
        print(f"Target FPS: {r.fps}")
    elif key == pygame.K_EQUALS or getattr(event, 'unicode', '') in ('+', '='):
        r.fps = min(120, r.fps + 5)
        print(f"Target FPS: {r.fps}")
    elif key == pygame.K_q:
        r._set_render_scale(max(0.25, r.render_scale - 0.25))
    elif key == pygame.K_e:
        r._set_render_scale(min(1.0, r.render_scale + 0.25))
    elif key == pygame.K_w:
        _cycle_wildness(r)
    elif key == pygame.K_r:
        _trigger_randomize(r)
    elif (key == pygame.K_9 or getattr(event, 'unicode', '') == '9') and not (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT)):
        _cycle_auto_random_bars(r)
    elif key == pygame.K_0 or getattr(event, 'unicode', '') == '0':
        _toggle_auto_random(r)
    elif key == pygame.K_g:
        r._drift_enabled = not r._drift_enabled
        print(f"Drift: {'ON' if r._drift_enabled else 'OFF'}")
    # Recording
    elif key == pygame.K_F9:
        r.toggle_recording()
    # MIDI learn
    elif key == pygame.K_m and r.midi and r.midi.connected:
        r.midi.learn_mode = not r.midi.learn_mode
        if r.midi.learn_mode:
            print("MIDI learn ON - 1-7=mode")
        else:
            print("MIDI learn OFF")
    elif r.midi and r.midi.learn_mode and pygame.K_1 <= key <= pygame.K_7:
        idx = key - pygame.K_1
        r.midi.learn_assign_note(MODE_NAMES[idx])
        r.midi.learn_mode = False
    # Presets
    elif pygame.K_1 <= key <= pygame.K_9 and event.mod & pygame.KMOD_CTRL:
        slot = str(key - pygame.K_1 + 1)
        r._presets[slot] = Preset(params=Params(**r.params.to_dict()), mode=r.mode)
        save_presets(r._presets, r._presets_path)
        print(f"Saved preset {slot}")
    elif pygame.K_1 <= key <= pygame.K_9 and event.mod & pygame.KMOD_ALT:
        slot = str(key - pygame.K_1 + 1)
        if slot in r._presets:
            p = r._presets[slot]
            r.params = Params(**p.params.to_dict())
            r.dp = derive(r.params)
            r.set_mode(p.mode)
            print(f"Loaded preset {slot}: mode={p.mode}")
        else:
            print(f"Preset {slot} is empty")
    # Mode selection
    elif pygame.K_1 <= key <= pygame.K_7:
        idx = key - pygame.K_1
        r.set_mode(MODE_NAMES[idx])
        print(f"Mode: {r.mode}")


AUTO_RANDOM_BAR_OPTIONS = [1, 2, 4, 8, 16, 32, "random"]


def _toggle_auto_random(r) -> None:
    r._auto_random = not r._auto_random
    if r._auto_random:
        r._auto_random_last_bar = -1
        print(f"Auto-random ON (every {r._auto_random_bars} bars)")
    else:
        print("Auto-random OFF")


def _cycle_auto_random_bars(r) -> None:
    idx = AUTO_RANDOM_BAR_OPTIONS.index(r._auto_random_bars) if r._auto_random_bars in AUTO_RANDOM_BAR_OPTIONS else 0
    r._auto_random_bars = AUTO_RANDOM_BAR_OPTIONS[(idx + 1) % len(AUTO_RANDOM_BAR_OPTIONS)]
    label = "random (4-32)" if r._auto_random_bars == "random" else f"{r._auto_random_bars} bars"
    print(f"Auto-random interval: {label}")


def check_auto_random(r, bar_idx: int) -> None:
    """Called each frame from the main loop. Triggers randomize on bar boundaries."""
    if not r._auto_random:
        return
    setting = r._auto_random_bars
    if setting == "random":
        bars = r._auto_random_next
    else:
        bars = setting
    # Trigger on multiples of N bars
    trigger_bar = (bar_idx // bars) * bars
    if trigger_bar != r._auto_random_last_bar and bar_idx > 0:
        r._auto_random_last_bar = trigger_bar
        if setting == "random":
            r._auto_random_next = random.choice([4, 8, 12, 16, 20, 24, 28, 32])
        _trigger_randomize(r)


WILDNESS_LEVELS = ["calm", "mild", "wild", "wilder", "any"]

_RANDOM_RANGES = {
    "calm": {
        "reactivity": (0.02, 0.12), "perception": (0.02, 0.1), "movement": (0.0, 0.08),
        "color": (0.02, 0.15), "brightness": (0.42, 0.52), "blend": (0.0, 0.1),
        "code": (0.0, 0.05), "img_blend": (0.0, 0.15), "zoom": (0.0, 0.1),
        "pan_x": (0.45, 0.55), "pan_y": (0.45, 0.55),
        "modes": ["feedback", "radial", "displacement"],
    },
    "mild": {
        "reactivity": (0.2, 0.5), "perception": (0.15, 0.45), "movement": (0.15, 0.4),
        "color": (0.2, 0.55), "brightness": (0.3, 0.6), "blend": (0.05, 0.4),
        "code": (0.0, 0.3), "img_blend": (0.05, 0.5), "zoom": (0.0, 0.4),
        "pan_x": (0.35, 0.65), "pan_y": (0.35, 0.65),
        "modes": ["feedback", "radial", "kaleidoscope", "displacement", "glitch"],
    },
    "wild": {
        "reactivity": (0.35, 0.7), "perception": (0.3, 0.65), "movement": (0.3, 0.65),
        "color": (0.35, 0.75), "brightness": (0.2, 0.7), "blend": (0.1, 0.7),
        "code": (0.1, 0.6), "img_blend": (0.1, 0.7), "zoom": (0.0, 0.6),
        "pan_x": (0.25, 0.75), "pan_y": (0.25, 0.75),
        "modes": MODE_NAMES,
    },
    "wilder": {
        "reactivity": (0.5, 1.0), "perception": (0.4, 1.0), "movement": (0.5, 1.0),
        "color": (0.5, 1.0), "brightness": (0.15, 0.85), "blend": (0.3, 1.0),
        "code": (0.3, 1.0), "img_blend": (0.3, 1.0), "zoom": (0.0, 0.8),
        "pan_x": (0.15, 0.85), "pan_y": (0.15, 0.85),
        "modes": ["glitch", "strobe", "kaleidoscope"],
    },
}


def _cycle_wildness(r) -> None:
    """Cycle wildness level: calm -> mild -> wild -> wilder -> any."""
    idx = WILDNESS_LEVELS.index(r._wildness) if r._wildness in WILDNESS_LEVELS else 0
    r._wildness = WILDNESS_LEVELS[(idx + 1) % len(WILDNESS_LEVELS)]
    print(f"Wildness: {r._wildness.upper()}")


def _trigger_randomize(r) -> None:
    """Randomize params at current wildness level."""
    if r._wildness == "any":
        _randomize_free(r)
    else:
        _randomize(r, r._wildness)


def _randomize(r, intensity: str) -> None:
    rng = _RANDOM_RANGES[intensity]
    r.params = Params(**{
        k: round(np.random.uniform(*rng[k]), 2)
        for k in rng if k != "modes"
    })
    r.dp = derive(r.params)
    r.set_mode(str(np.random.choice(rng["modes"])))
    print(f"RANDOM [{intensity.upper()}]: mode={r.mode}")


def _randomize_free(r) -> None:
    """Each param independently picks a random intensity level's range."""
    levels = list(_RANDOM_RANGES.values())
    params = {}
    for k in Params.__dataclass_fields__:
        rng = random.choice(levels)
        if k in rng and k != "modes":
            params[k] = round(np.random.uniform(*rng[k]), 2)
    r.params = Params(**params)
    r.dp = derive(r.params)
    mode_pool = random.choice(levels)["modes"]
    r.set_mode(str(np.random.choice(mode_pool)))
    print(f"RANDOM [ANY]: mode={r.mode}")
