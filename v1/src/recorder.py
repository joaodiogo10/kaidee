"""Live scene recorder — pipes frames to ffmpeg for MP4 output."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np


def _ffmpeg_exe() -> str:
    """Get the bundled ffmpeg binary from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Recorder:
    """Records live frames to MP4 via ffmpeg subprocess pipe.

    Frame writing happens on a background thread to avoid stalling
    the render loop with GPU readback latency and pipe I/O.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._video_path: str = ""
        self._final_path: str = ""
        self._w = 0
        self._h = 0
        self._fps = 60
        self._frame_count = 0
        self._wall_start = 0.0
        # Background writer
        self._queue: deque[bytes] = deque()
        self._writer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def recording(self) -> bool:
        return self._proc is not None

    def start(self, w: int, h: int, fps: int) -> str:
        """Start recording. Returns the output file path."""
        if self._proc is not None:
            return self._final_path

        os.makedirs("recordings", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._final_path = os.path.abspath(f"recordings/rec_{stamp}.mp4")
        self._video_path = os.path.abspath(f"recordings/.rec_{stamp}_tmp.mp4")
        self._w = w
        self._h = h
        self._fps = fps
        self._frame_count = 0
        self._wall_start = time.time()
        self._queue.clear()
        self._stop_event.clear()

        ffmpeg = _ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            self._video_path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        return self._final_path

    def _writer_loop(self) -> None:
        """Background thread: drain queue and write to ffmpeg stdin."""
        while not self._stop_event.is_set() or self._queue:
            if self._queue:
                data = self._queue.popleft()
                try:
                    if self._proc and self._proc.stdin:
                        self._proc.stdin.write(data)
                except BrokenPipeError:
                    break
            else:
                time.sleep(0.001)

    def write_frame(self, frame: np.ndarray) -> None:
        """Queue one (H, W, 3) uint8 RGB frame for writing."""
        if self._proc is None:
            return
        self._queue.append(np.ascontiguousarray(frame).tobytes())
        self._frame_count += 1

    def stop(self, audio_path: str | None = None,
             start_time: float = 0.0, end_time: float = 0.0) -> str | None:
        """Stop recording. Correct duration, mux audio. Returns final path."""
        if self._proc is None:
            return None

        # Drain the background writer
        self._stop_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=10.0)

        self._proc.stdin.close()
        self._proc.wait()
        self._proc = None
        self._writer_thread = None

        wall_dur = time.time() - self._wall_start

        if not os.path.exists(self._video_path):
            print("Recording failed — no video file produced")
            return None

        # Calculate duration correction.
        # Video was encoded at target fps, but actual frame rate was lower.
        # itsscale stretches timestamps so output duration matches wall clock.
        encoded_dur = self._frame_count / self._fps if self._fps > 0 else wall_dur
        itsscale = wall_dur / encoded_dur if encoded_dur > 0.01 else 1.0

        ffmpeg = _ffmpeg_exe()
        audio_abs = os.path.abspath(audio_path) if audio_path else None
        has_audio = audio_abs and os.path.exists(audio_abs)
        slice_audio = has_audio and end_time > start_time

        if has_audio and slice_audio:
            duration = end_time - start_time
            # Correct video speed + mux sliced audio (FileAudio)
            mux_cmd = [
                ffmpeg, "-y",
                "-itsscale", f"{itsscale:.6f}",
                "-i", self._video_path,
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-i", audio_abs,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                self._final_path,
            ]
        elif has_audio:
            # Correct video speed + mux full audio (live recording)
            mux_cmd = [
                ffmpeg, "-y",
                "-itsscale", f"{itsscale:.6f}",
                "-i", self._video_path,
                "-i", audio_abs,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                self._final_path,
            ]
        else:
            # No audio — just correct video speed
            mux_cmd = [
                ffmpeg, "-y",
                "-itsscale", f"{itsscale:.6f}",
                "-i", self._video_path,
                "-c:v", "copy",
                self._final_path,
            ]

        result = subprocess.run(mux_cmd, capture_output=True)
        if result.returncode == 0:
            os.remove(self._video_path)
        else:
            err = result.stderr.decode(errors='replace')[-300:]
            print(f"Warning: post-processing failed, keeping raw video\n  {err}")
            os.rename(self._video_path, self._final_path)

        actual_fps = self._frame_count / wall_dur if wall_dur > 0 else 0
        print(f"Recording saved: {self._final_path} "
              f"({self._frame_count} frames, {wall_dur:.1f}s, {actual_fps:.0f}fps actual)")
        return self._final_path
