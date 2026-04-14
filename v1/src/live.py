"""Kaidee Live — Realtime audio-reactive visuals.

Usage: python -m src.live [--play audio.mp3] [--device N] [--mode auto] [--fps 60] [--demo]
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import pygame
except ImportError:
    print("pygame not installed. Run: pip install pygame")
    sys.exit(1)

try:
    import moderngl as _mgl
except ImportError:
    _mgl = None

from src.audio import AudioAnalyzer, DemoAudio, FileAudio, list_devices as list_audio_devices, find_loopback_device
from src.assets import Assets
from src.params import Params, DerivedParams, Preset, derive, load_presets, save_presets
from src.drift import DriftState, DriftValues, DRIFT_DISABLED, compute_drift
from src.transforms import blend
from src.modes import MODE_CLASSES, MODE_NAMES, COMPLEMENTS, get_auto_mode, get_auto_mode_pool
from src.modes.base import Mode
from src.postprocess import compute_postfx
from src import hud as hud_module
from src import input as input_module
from src.recorder import Recorder

COMPILED_DIR = "compiled"
PRESETS_PATH = "presets.json"


class LiveRenderer:
    def __init__(self, assets: Assets, audio, fps: int = 60, midi=None):
        self.assets = assets
        self.audio = audio
        self.fps = fps
        self.midi = midi

        self._full_w, self._full_h = assets.resolution
        self.render_scale = 1.0
        self.w, self.h = self._full_w, self._full_h

        self.params = Params()
        self.dp = derive(self.params)
        self.mode = "auto"
        self.running = True
        self.fullscreen = False
        self.show_hud = True
        self.frame_count = 0
        self._wall_start = time.time()

        # Mode instances
        self._modes: dict[str, Mode] = {
            name: cls(self.w, self.h) for name, cls in MODE_CLASSES.items()
        }

        # Presets
        self._presets_path = PRESETS_PATH
        self._presets = load_presets(self._presets_path)

        # Drift
        self._drift_enabled = True
        self._drift_state = DriftState()
        self._set_duration = getattr(audio, 'duration', 300.0)
        self._current_drift = DRIFT_DISABLED

        # CDJ-style seeking
        self._seeking = False
        self._seek_hold_start = 0.0
        self._seek_pos = 0.0
        self._seek_was_paused = False

        # GPU rotation state
        self._global_rot_angle = 0.0
        self._last_uniform_time = 0.0

        # HUD dirty-flag caching
        self._hud_dirty = True
        self._hud_last_state = None

        # Recording
        self.recorder = Recorder()
        self._record_start_time = 0.0
        self._current_t = 0.0

        # Randomize state (mutated by input module)
        self._wildness = "mild"
        self._auto_random = False
        self._auto_random_bars: int | str = 8
        self._auto_random_last_bar = -1
        self._auto_random_next = 4

        # Track screen size for GPU viewport updates (fullscreen, multi-monitor)
        self._last_screen_size = (0, 0)

        # Pygame setup
        pygame.init()
        pygame.display.set_caption("Kaidee Live")
        if _mgl is None:
            print("ERROR: moderngl is required for GPU rendering.")
            print("  Install it with: pip install -e '.[live]'")
            sys.exit(1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
        self.screen = pygame.display.set_mode(
            (self._full_w, self._full_h), pygame.OPENGL | pygame.DOUBLEBUF
        )
        self.gpu = None
        self._init_gpu()
        self._gpu_cpu_skip = 2
        self._set_render_scale(0.75)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 13)

    def _init_gpu(self):
        from src.gpu import GPUPipeline
        self.gpu = None
        self._hud_last_state = None  # invalidate HUD cache
        self.gpu = GPUPipeline(self.w, self.h, self._full_w, self._full_h, self.assets.vignette)

    def _set_render_scale(self, scale):
        self.render_scale = scale
        self.w = int(self._full_w * scale)
        self.h = int(self._full_h * scale)
        self.assets.rescale(self._full_w, self._full_h, scale)
        for mode in self._modes.values():
            mode.reset(self.w, self.h)
        if self.gpu:
            self.gpu.recreate_textures(self.w, self.h, self.assets.vignette)
        print(f"Render scale: {scale:.0%} ({self.w}x{self.h})")

    def set_mode(self, mode: str):
        self.mode = mode
        for m in self._modes.values():
            m.reset(self.w, self.h)

    def toggle_recording(self):
        """Start or stop recording the live scene to MP4."""
        t = self._current_t
        if self.recorder.recording:
            # Get audio: FileAudio has a filepath, AudioAnalyzer records live
            audio_path = getattr(self.audio, 'filepath', None)
            rec_audio = None
            if audio_path is None and hasattr(self.audio, 'stop_recording'):
                rec_audio = self.audio.stop_recording()
            if rec_audio:
                path = self.recorder.stop(rec_audio, 0.0, 0.0)
                try:
                    os.remove(rec_audio)
                except OSError:
                    pass
            else:
                path = self.recorder.stop(audio_path, self._record_start_time, t)
            if path:
                dur = t - self._record_start_time
                print(f"Recording stopped ({dur:.1f}s) -> {path}")
        else:
            rec_w = self._full_w
            rec_h = self._full_h
            path = self.recorder.start(rec_w, rec_h, self.fps)
            self._record_start_time = t
            if hasattr(self.audio, 'start_recording'):
                self.audio.start_recording()
            print(f"Recording started -> {path}")

    def _get_audio(self):
        dp = self.dp
        return (
            min(1.0, self.audio.bass * dp.bass_gain),
            min(1.0, self.audio.mids * dp.mids_gain),
            min(1.0, self.audio.highs * dp.highs_gain),
            min(1.0, self.audio.energy * (dp.bass_gain + dp.mids_gain + dp.highs_gain) / 3),
            min(1.0, self.audio.onset * (1.0 + dp.bass_gain * 0.2)),
        )

    # ---- Image blending (passed to modes) ----

    def _frame_key(self, t: float, variant: str) -> str | None:
        drift = self._current_drift
        bar_idx = drift.bar_idx
        if self.assets.n_images > 0:
            i = (bar_idx // 4) % self.assets.n_images
            return f"img{i}_{variant}"
        code_imgs = self.assets.code_images
        if code_imgs:
            idx = (bar_idx // 2) % len(code_imgs)
            return f"{code_imgs[idx]}_{variant}"
        return None

    def _img_blend(self, t: float, variant: str) -> np.ndarray | None:
        # Per-frame cache — avoids recomputing same variant across
        # primary mode, warp, and secondary mode within one frame.
        cache = getattr(self, '_img_cache', None)
        cache_t = getattr(self, '_img_cache_t', -1.0)
        if cache is not None and cache_t == t and variant in cache:
            return cache[variant]
        if cache_t != t:
            cache = {}
            self._img_cache = cache
            self._img_cache_t = t

        result = self._img_blend_inner(t, variant)
        cache[variant] = result
        return result

    def _img_blend_inner(self, t: float, variant: str) -> np.ndarray | None:
        drift = self._current_drift
        bar_idx = drift.bar_idx
        bar_phase = drift.bar_phase
        dp = self.dp
        n_img = self.assets.n_images

        if n_img == 0:
            code_imgs = self.assets.code_images
            if not code_imgs:
                return np.zeros((self.h, self.w, 3), np.uint8)
            if len(code_imgs) == 1:
                return self.assets.get(f"{code_imgs[0]}_{variant}") or np.zeros((self.h, self.w, 3), np.uint8)
            section = bar_idx // 2
            idx = section % len(code_imgs)
            nxt = (idx + 1) % len(code_imgs)
            frac = (bar_idx % 2 + bar_phase) / 2
            a = self.assets.get(f"{code_imgs[idx]}_{variant}")
            b = self.assets.get(f"{code_imgs[nxt]}_{variant}")
            if a is None or b is None:
                return a if a is not None else (b or np.zeros((self.h, self.w, 3), np.uint8))
            return self._crossfade(a, b, frac, dp.img_crossfade)

        cyc = dp.img_cycle_bars
        section = bar_idx // cyc
        i = section % n_img
        j = (i + 1) % n_img
        frac = (bar_idx % cyc + bar_phase) / cyc
        a = self.assets.get(f"img{i}_{variant}")
        b = self.assets.get(f"img{j}_{variant}")
        if a is None or b is None:
            return a if a is not None else b

        result = self._crossfade(a, b, frac, dp.img_crossfade)

        # Extra overlap from perception
        if dp.overlap > 0.01 and n_img > 2:
            bass_g = min(1.0, self.audio.bass * dp.bass_gain)
            mids_g = min(1.0, self.audio.mids * dp.mids_gain)
            k = (i + n_img // 2) % n_img
            alt_variants = ["deep", "crot_120", "crot_240", "solar_80"]
            alt_v = alt_variants[int(bar_idx * 0.5) % len(alt_variants)]
            c = self.assets.get(f"img{k}_{alt_v}")
            if c is not None:
                overlap_alpha = dp.overlap * (0.15 + bass_g * 0.25 + mids_g * 0.15)
                result = blend(result, c, min(overlap_alpha, 0.6))

        return result

    def _crossfade(self, a, b, frac, cf):
        hold_start = 0.5 - cf * 0.5
        hold_end = 0.5 + cf * 0.5
        if frac < hold_start:
            return a
        if frac > hold_end:
            return b
        if hold_end - hold_start < 0.01:
            return a if frac < 0.5 else b
        t_blend = (frac - hold_start) / (hold_end - hold_start)
        s = t_blend * t_blend * (3 - 2 * t_blend)
        return blend(a, b, s)

    # ---- Code overlay ----

    _ONSET_VARIANTS = ["sharp", "sorted", "inverted", "poster_2"]
    _ENERGY_VARIANTS = ["abstract", "sorted", "solar_80", "inverted"]
    _SHARP_VARIANTS = ["sharp", "sorted", "sharp", "deep"]

    def _code_img(self, t, variant):
        code_imgs = self.assets.code_images
        if not code_imgs:
            return None
        bar_idx = self._current_drift.bar_idx
        idx = bar_idx % len(code_imgs)
        name = code_imgs[idx]
        return self.assets.get(f"{name}_{variant}")

    def _select_code_overlay(self, t, bass, energy, onset, code_level=0.0):
        # Lower audio thresholds proportionally to code_level
        bass_thresh = max(0.1, 0.7 - code_level * 0.5)
        onset_thresh = max(0.05, 0.3 - code_level * 0.2)
        energy_thresh = max(0.05, 0.2 - code_level * 0.15)
        use_sharp = code_level > 0.5
        if bass > bass_thresh:
            variant = "sharp" if use_sharp else "inverted"
            return self._code_img(t, variant), (bass - bass_thresh + 0.2) * 0.8
        if onset > onset_thresh:
            variants = self._SHARP_VARIANTS if use_sharp else self._ONSET_VARIANTS
            variant = variants[int(t / 2) % len(variants)]
            return self._code_img(t, variant), onset * 0.7
        if energy > energy_thresh:
            variants = self._SHARP_VARIANTS if use_sharp else self._ENERGY_VARIANTS
            variant = variants[int(t / 3) % len(variants)]
            return self._code_img(t, variant), (energy - energy_thresh + 0.1) * 0.5
        return None, 0.0

    # ---- Dispatch ----

    def _dispatch(self, mode_name: str, t: float) -> np.ndarray:
        mode = self._modes.get(mode_name)
        if mode is None:
            return self._img_blend(t, "abstract") or np.zeros((self.h, self.w, 3), np.uint8)

        bass, mids, highs, energy, onset = self._get_audio()
        return mode.render(
            t, bass, mids, highs, energy, onset,
            self.dp, self._current_drift, self.assets,
            self._img_blend, self._frame_key, self.fps,
        )

    # ---- Main render ----

    def _render_frame(self, t: float):
        _p = time.perf_counter
        _r0 = _p()

        bass, mids, highs, energy, onset = self._get_audio()
        drift = self._current_drift
        dp = self.dp

        mode = self.mode
        if mode == "auto":
            bar_idx = drift.bar_idx
            bar_phase = drift.bar_phase
            pool = get_auto_mode_pool(drift.mode_energy)
            mode = get_auto_mode(drift.mode_energy, bar_idx)

            # Crossfade during last beat of 4-bar phrase
            if bar_idx % 4 == 3 and bar_phase > 0.75:
                next_section = bar_idx // 4 + 1
                next_mode = pool[next_section % len(pool)]
                fade = (bar_phase - 0.75) / 0.25
                a = self._dispatch(mode, t)
                b = self._dispatch(next_mode, t)
                frame = blend(a, b, fade)
            else:
                frame = self._dispatch(mode, t)
        else:
            frame = self._dispatch(mode, t)

        _r1 = _p()

        # Image superimposition
        n_img = self.assets.n_images
        if dp.img_layers > 1 and n_img > 2:
            bar_idx_il = drift.bar_idx
            cyc_il = dp.img_cycle_bars
            section_il = bar_idx_il // cyc_il
            i_il = section_il % n_img
            layer_variants = ["deep", "crot_120", "solar_80", "abstract"]
            for layer_n in range(1, dp.img_layers):
                k = (i_il + layer_n * max(1, n_img // dp.img_layers)) % n_img
                if k == i_il:
                    k = (k + 1) % n_img
                alt_v = layer_variants[(layer_n - 1) % len(layer_variants)]
                c = self.assets.get(f"img{k}_{alt_v}")
                if c is not None:
                    frame = blend(frame, c, min(dp.img_layer_alpha, 0.75))

        # Warp
        if dp.warp < 1.0:
            sharp = self._img_blend(t, "sharp")
            if sharp is not None:
                frame = blend(sharp, frame, dp.warp)

        _r2 = _p()

        # Code overlay + displacement
        if self.assets.code_images and dp.code_vis > 0.01:
            code_level = dp.code_blend ** 0.5  # base alpha from code param
            code_frame, code_alpha = self._select_code_overlay(t, bass, energy, onset, code_level)
            # At high code values, bypass drift dampening
            drift_factor = drift.code_mix + (1.0 - drift.code_mix) * dp.code_blend
            code_alpha *= drift_factor * dp.code_blend
            if code_frame is not None and code_alpha > 0.02:
                # Scroll code vertically — speed driven by energy
                scroll_speed = 30 + energy * 120  # px/sec
                scroll_y = int(t * scroll_speed) % self.h
                if scroll_y > 0:
                    code_frame = np.roll(code_frame, -scroll_y, axis=0)
                # Displacement on the base frame
                disp_scale = 1.0 - dp.code_blend * 0.7
                disp_str = dp.code_vis * dp.react_mod * bass * 15 * disp_scale
                if disp_str > 1.0:
                    blk = 16
                    n_blocks = self.h // blk
                    trim_h = n_blocks * blk
                    lum = np.mean(code_frame[:trim_h].astype(np.float32), axis=2)
                    block_means = lum.reshape(n_blocks, blk, self.w).mean(axis=(1, 2))
                    shifts = (block_means / 255.0 * disp_str - disp_str * 0.5).astype(int)
                    x_base = np.arange(self.w)
                    for i, s in enumerate(shifts):
                        if s != 0:
                            b0, b1 = i * blk, (i + 1) * blk
                            frame[b0:b1] = frame[b0:b1][:, (x_base - s) % self.w]
                rgb_sep = int(dp.code_vis * dp.react_mod * energy * 8 * disp_scale)
                if rgb_sep > 1:
                    frame[:, :, 0] = np.roll(frame[:, :, 0], rgb_sep, axis=1)
                    frame[:, :, 2] = np.roll(frame[:, :, 2], -rgb_sep, axis=1)
                # Pulse alpha with bass
                beat_pulse = max(0, 1.0 - drift.beat_phase * 4) ** 2
                pulse_alpha = code_alpha * (0.7 + 0.3 * beat_pulse)
                frame = blend(frame, code_frame, min(pulse_alpha, 0.95))

        _r3 = _p()

        # Ghost image overlay
        if dp.overlap > 0.01 and n_img > 2:
            ghost_idx = (drift.bar_idx // dp.img_cycle_bars + 2) % n_img
            ghost_variant = "abstract" if energy > 0.4 else "deep"
            ghost = self.assets.get(f"img{ghost_idx}_{ghost_variant}")
            if ghost is not None:
                ghost_alpha = dp.overlap * (0.1 + bass * 0.2 + mids * 0.15)
                frame = blend(frame, ghost, min(ghost_alpha, 0.5))

        _r4 = _p()
        result = self._gpu_postprocess(frame, t, mode, bass, mids, energy, drift)
        _r5 = _p()

        # Store sub-timings for periodic reporting
        self._render_sub = (
            (_r1 - _r0) * 1000,  # dispatch (mode render)
            (_r2 - _r1) * 1000,  # img layers + warp
            (_r3 - _r2) * 1000,  # code overlay
            (_r4 - _r3) * 1000,  # ghost
            (_r5 - _r4) * 1000,  # gpu postprocess
        )
        return result

    def _gpu_postprocess(self, frame, t, mode, bass, mids, energy, drift):
        dp = self.dp

        # Secondary mode blend — expensive (full mode dispatch), so only
        # update every 4th CPU frame and reuse the cached texture otherwise.
        self.gpu.cached_has_secondary = False
        if self.params.blend > 0.05:
            self._secondary_age = getattr(self, '_secondary_age', 0) + 1
            if self._secondary_age >= 4 or not getattr(self, '_has_cached_secondary', False):
                secondary_mode = COMPLEMENTS.get(mode, "kaleidoscope")
                secondary = self._dispatch(secondary_mode, t)
                self.gpu.upload_secondary(secondary)
                self._secondary_age = 0
                self._has_cached_secondary = True
            else:
                # Reuse the texture already on GPU
                self.gpu.cached_has_secondary = True

        self.gpu.upload_frame(frame)
        uniforms = self._compute_gpu_uniforms(t)
        self.gpu.render(uniforms)
        return None

    def _gpu_rerender(self, t):
        drift = self._compute_drift(t)
        self._current_drift = drift
        uniforms = self._compute_gpu_uniforms(t)
        self.gpu.render(uniforms)

    def _compute_gpu_uniforms(self, t):
        bass, mids, highs, energy, onset = self._get_audio()
        now = time.time()
        dt = min(now - self._last_uniform_time, 0.1) if self._last_uniform_time > 0 else 1.0 / self.fps
        self._last_uniform_time = now
        fx, self._global_rot_angle = compute_postfx(
            t, dt, self.dp, self._current_drift, bass, energy,
            self.params, self._global_rot_angle)

        # GPU trail state tracking
        if fx.trail_keep > 0.01:
            self.gpu.trail_active = True
        elif self.gpu.trail_active:
            self.gpu.trail_active = False

        has_secondary = self.gpu.cached_has_secondary

        return {
            'u_zoom': float(fx.zoom),
            'u_pan': (float(fx.pan_x), float(fx.pan_y)),
            'u_rot_shift': (fx.rot_shift_x, fx.rot_shift_y),
            'u_saturation': float(fx.saturation),
            'u_contrast': float(fx.contrast),
            'u_hue_shift': (fx.hue_r, fx.hue_g, fx.hue_b),
            'u_brightness': float(fx.brightness),
            'u_chroma': float(fx.chroma_h / self.w) if fx.chroma_h else 0.0,
            'u_chroma_v': float(fx.chroma_v / self.h) if fx.chroma_v else 0.0,
            'u_flash': float(fx.flash),
            'u_chaos': float(fx.chaos),
            'u_chaos_spacing': fx.chaos_spacing,
            'u_blend_alpha': float(fx.blend_alpha) if has_secondary else 0.0,
            'u_trail_keep': float(fx.trail_keep),
            'u_vig_strength': float(fx.vig_strength),
            'u_has_secondary': 1 if has_secondary else 0,
            'u_has_trail': 1 if self.gpu.trail_active else 0,
            'u_time': float(t),
        }

    # ---- Drift ----

    def _compute_drift(self, t):
        if not self._drift_enabled:
            bass, mids, highs, energy, onset = self._get_audio()
            d = DRIFT_DISABLED
            # Still compute bar/beat info
            bar_idx, bar_phase = self.audio.get_bar(t)
            _, beat_phase = self.audio.get_beat_phase(t)
            bpm = self.audio.get_bpm(t)
            return DriftValues(
                intensity=1.0, warmth=0.0, code_mix=0.7, chaos=0.0,
                vignette=0.75, mode_energy=0.5, time_arc=0.5, music_intensity=0.5,
                bpm=bpm, bar_idx=bar_idx, bar_phase=bar_phase, beat_phase=beat_phase,
            )
        bass, mids, highs, energy, onset = self._get_audio()
        return compute_drift(
            t, self.fps, self._set_duration,
            bass, energy, onset, self.audio, self._drift_state,
        )

    # ---- Main loop ----

    def run(self):
        self.audio.start()
        # Timing diagnostics
        _perf = time.perf_counter
        _t_accum = {'events': 0.0, 'audio': 0.0, 'drift': 0.0,
                     'render': 0.0, 'record': 0.0, 'hud+flip': 0.0,
                     'tick': 0.0, 'total': 0.0}
        _t_sub = {'dispatch': 0.0, 'layers': 0.0, 'code': 0.0,
                  'ghost': 0.0, 'gpu_post': 0.0}
        _t_frames = 0
        _t_render_frames = 0
        self._render_sub = None
        try:
            while self.running:
                _t0 = _perf()

                self._handle_events()

                # MIDI
                if self.midi and self.midi.connected:
                    events = self.midi.poll()
                    if not self.midi.learn_mode:
                        changed = self.midi.apply_events(events, self.params, self.set_mode)
                        if changed:
                            self.dp = derive(self.params)

                # CDJ seeking
                if isinstance(self.audio, FileAudio):
                    keys = pygame.key.get_pressed()
                    holding_seek = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
                    if holding_seek:
                        direction = -1 if keys[pygame.K_LEFT] else 1
                        now = time.time()
                        if not self._seeking:
                            self._seeking = True
                            self._seek_hold_start = now
                            self._seek_pos = self.audio.get_time()
                            self._seek_was_paused = self.audio.paused
                            if not self.audio.paused:
                                pygame.mixer.music.pause()
                        hold_dur = now - self._seek_hold_start
                        speed = 30.0 * (18.0 ** min(hold_dur, 2))
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            speed *= 4
                        self._seek_pos += direction * speed / self.fps
                        self._seek_pos = max(0.0, min(self._seek_pos, self.audio.duration - 0.1))
                    elif self._seeking:
                        self._seeking = False
                        self.audio.seek(self._seek_pos)
                        if self._seek_was_paused:
                            pygame.mixer.music.pause()
                            self.audio.paused = True

                _t1 = _perf()

                # Time
                if isinstance(self.audio, FileAudio):
                    t = self._seek_pos if self._seeking else self.audio.get_time()
                    self.audio.update(t)
                elif isinstance(self.audio, DemoAudio):
                    t = self.frame_count / self.fps
                    self.audio.update(t)
                else:
                    t = time.time() - self._wall_start
                    self.audio.update(t)

                _t2 = _perf()

                # Drift
                self._current_t = t
                self._current_drift = self._compute_drift(t)

                # Auto-randomize
                input_module.check_auto_random(self, self._current_drift.bar_idx)

                _t3 = _perf()

                # Update viewport if window size changed (fullscreen toggle, etc)
                # get_window_size() queries the actual SDL window, unlike
                # screen.get_size() which returns the stale OpenGL surface size
                screen_size = pygame.display.get_window_size()
                if screen_size != self._last_screen_size:
                    self._last_screen_size = screen_size
                    self.gpu.set_screen_size(*screen_size)

                gpu_skip = self._gpu_cpu_skip
                if self.frame_count % gpu_skip == 0:
                    self._render_frame(t)
                else:
                    self._gpu_rerender(t)

                _t4 = _perf()

                # Capture frame for recording BEFORE HUD overlay
                if self.recorder.recording:
                    self.recorder.write_frame(self.gpu.readback())

                _t5 = _perf()

                self._display_gpu_hud(t)

                _t6 = _perf()

                actual_fps = self.clock.get_fps()
                if self.frame_count > 10 and self.frame_count % 30 == 0:
                    if actual_fps < self.fps * 0.6 and gpu_skip < 4:
                        self._gpu_cpu_skip = gpu_skip + 1
                    elif actual_fps > self.fps * 0.85 and gpu_skip > 1:
                        self._gpu_cpu_skip = gpu_skip - 1

                self.clock.tick(self.fps)

                _t7 = _perf()

                # Accumulate timing
                _t_accum['events'] += _t1 - _t0
                _t_accum['audio'] += _t2 - _t1
                _t_accum['drift'] += _t3 - _t2
                _t_accum['render'] += _t4 - _t3
                _t_accum['record'] += _t5 - _t4
                _t_accum['hud+flip'] += _t6 - _t5
                _t_accum['tick'] += _t7 - _t6
                _t_accum['total'] += _t7 - _t0
                _t_frames += 1
                if self._render_sub is not None:
                    _t_sub['dispatch'] += self._render_sub[0]
                    _t_sub['layers'] += self._render_sub[1]
                    _t_sub['code'] += self._render_sub[2]
                    _t_sub['ghost'] += self._render_sub[3]
                    _t_sub['gpu_post'] += self._render_sub[4]
                    _t_render_frames += 1
                    self._render_sub = None

                if _t_frames == 60:
                    parts = '  '.join(
                        f"{k}:{v/_t_frames*1000:.1f}ms"
                        for k, v in _t_accum.items()
                    )
                    print(f"[perf] {parts}  fps:{actual_fps:.0f}")
                    if _t_render_frames > 0:
                        sub = '  '.join(
                            f"{k}:{v/_t_render_frames:.1f}ms"
                            for k, v in _t_sub.items()
                        )
                        print(f"  [render] {sub}  ({_t_render_frames} cpu frames / 60)")
                    for k in _t_accum:
                        _t_accum[k] = 0.0
                    for k in _t_sub:
                        _t_sub[k] = 0.0
                    _t_frames = 0
                    _t_render_frames = 0

                self.frame_count += 1
        finally:
            if self.recorder.recording:
                audio_path = getattr(self.audio, 'filepath', None)
                rec_audio = None
                if audio_path is None and hasattr(self.audio, 'stop_recording'):
                    rec_audio = self.audio.stop_recording()
                if rec_audio:
                    self.recorder.stop(rec_audio, 0.0, 0.0)
                    try:
                        os.remove(rec_audio)
                    except OSError:
                        pass
                else:
                    self.recorder.stop(audio_path, self._record_start_time, self._current_t)
            self.audio.stop()
            if self.midi:
                self.midi.close()
            if self.gpu:
                self.gpu.release()
            pygame.quit()

    # ---- Display ----

    def _display_gpu_hud(self, t):
        if self.show_hud and self.gpu:
            # HUD renders at actual screen resolution for crisp text
            hud_w, hud_h = self.gpu.tex_hud.size
            bass, mids, highs, _, _ = self._get_audio()
            hud_state = (
                int(t), int(self.clock.get_fps()), self.fps, self.mode,
                self.w, self.h, self._gpu_cpu_skip,
                round(self.params.reactivity, 2), round(self.params.perception, 2),
                round(self.params.movement, 2), round(self.params.color, 2),
                round(self.params.brightness, 2), round(self.params.blend, 2),
                round(self.params.img_blend, 2),
                round(self.params.zoom, 2),
                round(self.params.pan_x, 2), round(self.params.pan_y, 2),
                int(bass * 5), int(mids * 5), int(highs * 5),
                self._seeking, self._drift_enabled,
                self.recorder.recording,
                getattr(self, '_auto_random', False),
                getattr(self, '_auto_random_bars', 8),
                hud_w, hud_h,
            )
            if hud_state != self._hud_last_state:
                self._hud_last_state = hud_state
                # Cache font — only recreate if size changes
                scale = hud_h / self._full_h
                font_size = max(10, int(13 * scale))
                if font_size != getattr(self, '_hud_font_size', 0):
                    self._hud_font_size = font_size
                    self._hud_font = pygame.font.SysFont("monospace", font_size)
                hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
                hud_surface.fill((0, 0, 0, 0))
                hud_module.render_hud_text(
                    hud_surface, self._hud_font, t,
                    self.clock.get_fps(), self.fps, self.mode,
                    self.w, self.h, self._gpu_cpu_skip,
                    self.params, self.dp, self._current_drift, self._drift_enabled,
                    self.audio, bass, mids, highs, self.midi, self._seeking,
                    recording=self.recorder.recording,
                    wildness=getattr(self, '_wildness', 'mild'),
                    auto_random=getattr(self, '_auto_random', False),
                    auto_random_bars=getattr(self, '_auto_random_bars', 8),
                )
                hud_arr = pygame.surfarray.array3d(hud_surface).swapaxes(0, 1)
                hud_alpha = pygame.surfarray.array_alpha(hud_surface).swapaxes(0, 1)
                hud_rgba = np.zeros((hud_h, hud_w, 4), dtype=np.uint8)
                hud_rgba[:, :, :3] = hud_arr
                hud_rgba[:, :, 3] = hud_alpha
                self.gpu.upload_hud(hud_rgba)
            self.gpu.render_hud_overlay()
        pygame.display.flip()

    # ---- Event handling (delegated to input module) ----

    def _handle_events(self):
        input_module.handle_events(self)


def main():
    parser = argparse.ArgumentParser(description="Kaidee Live - Realtime visuals")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--mode", default="auto", choices=MODE_NAMES)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--play", type=str, default=None, metavar="FILE",
                        help="Play audio file and sync visuals")
    parser.add_argument("--demo", action="store_true", help="Use demo audio (no mic)")
    parser.add_argument("--loopback", action="store_true",
                        help="Capture system audio via loopback device (BlackHole, etc.)")
    parser.add_argument("--midi", type=str, default=None, const="", nargs="?", metavar="NAME",
                        help="Enable MIDI controller")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--list-midi", action="store_true", help="List MIDI devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.list_midi:
        from src.midi import MidiController
        MidiController.list_devices()
        return

    assets = Assets(COMPILED_DIR)

    # Audio
    from src.audio import sd
    if args.play:
        if not os.path.exists(args.play):
            print(f"ERROR: Audio file not found: {args.play}")
            sys.exit(1)
        audio = FileAudio(args.play)
    elif args.loopback:
        dev_idx = find_loopback_device()
        if dev_idx is not None:
            audio = AudioAnalyzer(device=dev_idx)
            if audio._stream is None:
                print("ERROR: Could not open loopback device.")
                sys.exit(1)
        else:
            print("ERROR: No loopback audio device found.")
            print("  Install a virtual audio loopback driver:")
            print("    macOS:   brew install blackhole-2ch")
            print("    Linux:   PulseAudio/PipeWire monitor sources work automatically")
            print("    Windows: Install VB-Cable (https://vb-audio.com/Cable/)")
            print(f"\n  Available devices (use --list-devices to see all):")
            if sd is not None:
                for idx, dev in enumerate(sd.query_devices()):
                    if dev["max_input_channels"] > 0:
                        print(f"    [{idx}] {dev['name']}")
            sys.exit(1)
    elif args.demo or sd is None:
        audio = DemoAudio()
        print("Using demo audio")
    else:
        audio = AudioAnalyzer(device=args.device)
        if audio._stream is None:
            audio = DemoAudio()
            print("Falling back to demo audio")

    # MIDI
    midi = None
    if args.midi is not None:
        from src.midi import MidiController
        device_name = args.midi if args.midi else None
        midi = MidiController(device_name=device_name)
        if not midi.connected:
            print("No MIDI device found.")
            midi = None

    renderer = LiveRenderer(assets, audio, fps=args.fps, midi=midi)
    renderer.set_mode(args.mode)
    print(f"\nStarting live renderer - mode: {args.mode}, fps: {args.fps}")
    print("Press H to toggle HUD, F for fullscreen, Q to quit\n")
    renderer.run()


if __name__ == "__main__":
    main()
