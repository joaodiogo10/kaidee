"""HUD overlay rendering."""

from __future__ import annotations

import pygame

from src.audio import FileAudio
from src.params import Params, DerivedParams
from src.drift import DriftValues
from src.midi import MidiController
from src.modes import get_auto_mode


def render_hud_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    t: float,
    fps_actual: float,
    fps_target: int,
    mode: str,
    render_w: int,
    render_h: int,
    gpu_skip: int,
    params: Params,
    dp: DerivedParams,
    drift: DriftValues,
    drift_enabled: bool,
    audio,
    bass: float,
    mids: float,
    highs: float,
    midi: MidiController | None,
    seeking: bool,
    recording: bool = False,
    wildness: str = "mild",
    auto_random: bool = False,
    auto_random_bars: int | str = 8,
) -> None:
    """Render HUD text onto a pygame surface."""
    mode_display = mode
    if mode == "auto":
        active = get_auto_mode(drift.mode_energy, drift.bar_idx)
        mode_display = f"auto -> {active}"

    drift_info = ""
    if drift_enabled:
        drift_info = (f"  arc:{drift.time_arc:.1f}"
                      f" mus:{drift.music_intensity:.1f}"
                      f" bpm:{drift.bpm:.0f}"
                      f" bar:{drift.bar_idx}"
                      f" chaos:{drift.chaos:.1f}")

    time_line = ""
    if isinstance(audio, FileAudio):
        dur = audio.duration
        pos = min(t, dur)
        pct = pos / dur if dur > 0 else 0
        bar_len = 30
        filled = int(pct * bar_len)
        time_bar = '=' * filled + '>' + '-' * (bar_len - filled - 1)
        vol = audio.get_volume()
        m_pos, s_pos = int(pos) // 60, int(pos) % 60
        m_dur, s_dur = int(dur) // 60, int(dur) % 60
        seek_tag = " <<SEEK>>" if seeking else ""
        time_line = f"  {m_pos}:{s_pos:02d}/{m_dur}:{s_dur:02d} [{time_bar}] Vol:{vol:.0%}{seek_tag}"

    def bar(val):
        return '#' * int(val * 20)

    scale_info = f"{render_w}x{render_h} skip:{gpu_skip}"

    lines = [
        f"FPS: {fps_actual:.0f}/{fps_target}  {scale_info}  Mode: {mode_display}{time_line}",
        f"REACT: {bar(params.reactivity):20s} {params.reactivity:.2f} [Z/X]   MOVE:  {bar(params.movement):20s} {params.movement:.2f} [A/S]",
        f"PERC:  {bar(params.perception):20s} {params.perception:.2f} [,/.]   COLOR: {bar(params.color):20s} {params.color:.2f} [B/N]",
        f"BRIGHT:{bar(params.brightness):20s} {params.brightness:.2f} [J/K]   BLEND: {bar(params.blend):20s} {params.blend:.2f} [C/V]",
        f"CODE:  {bar(params.code):20s} {params.code:.2f} [</>]   ZOOM:  {bar(params.zoom):20s} {params.zoom:.2f} [[/]]",
        f"IMGBL: {bar(params.img_blend):20s} {params.img_blend:.2f} [U/I]",
        f"PAN X: {bar(params.pan_x):20s} {params.pan_x:.2f} [T/Y]   PAN Y: {bar(params.pan_y):20s} {params.pan_y:.2f} [O/P]",
        f"Bass: {bar(bass):20s}  Mids: {bar(mids):20s}  High: {bar(highs):20s}",
        f"[1-7] modes  [W] wildness:{wildness}  [R] randomize  [9] interval:{'rnd' if auto_random_bars == 'random' else str(auto_random_bars)}bars [0] auto:{'ON' if auto_random else 'OFF'}  [H] hide{drift_info}",
    ]

    if midi and midi.connected:
        if midi.learn_mode:
            midi_line = ">>> MIDI LEARN: move knob, then 1-7=mode <<<"
        else:
            maps = [f"CC{cc}={p}" for cc, p in sorted(midi.cc_map.items())]
            midi_line = f"MIDI: {' | '.join(maps)}" if maps else "MIDI: move knobs to auto-map"
            if midi._status and "Auto:" in midi._status:
                midi_line += f"  ({midi._status})"
        lines.append(midi_line)

    line_h = font.get_linesize()
    margin = max(10, line_h // 2)
    for j, line in enumerate(lines):
        surf = font.render(line, True, (0, 255, 0))
        surface.blit(surf, (margin, margin + j * line_h))

    if recording:
        rec_surf = font.render("  REC  [F9]", True, (255, 60, 60))
        surface.blit(rec_surf, (surface.get_width() - rec_surf.get_width() - margin, margin))
