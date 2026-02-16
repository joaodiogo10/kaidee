"""MIDI controller input with auto-mapping."""

from __future__ import annotations
import sys

try:
    import mido
except ImportError:
    mido = None

from src.modes import MODE_NAMES

PARAM_ORDER = ["reactivity", "perception", "movement", "color",
               "brightness", "blend", "img_blend"]


class MidiController:
    """MIDI input with auto-mapping.

    Auto-map: first 7 unique knobs (CCs) → params in order.
    First 7 unique pads (notes) → modes 1-7.
    Press M to enter manual learn mode.
    """

    def __init__(self, device_name: str | None = None):
        if mido is None:
            print("ERROR: mido not installed. Run: pip install mido python-rtmidi")
            sys.exit(1)

        self._port = None
        self.cc_map: dict[int, str] = {}
        self.note_map: dict[int, str] = {}
        self.learn_mode = False
        self._last_cc: int | None = None
        self._last_note: int | None = None
        self._status = ""
        self._pending: list[tuple[str, int, int]] = []
        self._auto_cc_order: list[int] = []
        self._auto_note_order: list[int] = []

        if device_name is None:
            device_name = self._find_input()
        if device_name is not None:
            try:
                self._port = mido.open_input(device_name, callback=self._on_message)
                print(f"MIDI connected: {device_name}")
                self._status = f"MIDI: {device_name}"
            except Exception as e:
                print(f"WARNING: Could not open MIDI device '{device_name}': {e}")

    def _on_message(self, msg) -> None:
        if msg.type == "control_change":
            self._last_cc = msg.control
            if msg.control not in self.cc_map and msg.control not in self._auto_cc_order:
                self._auto_cc_order.append(msg.control)
                if len(self._auto_cc_order) <= len(PARAM_ORDER):
                    param = PARAM_ORDER[len(self._auto_cc_order) - 1]
                    self.cc_map[msg.control] = param
                    self._status = f"Auto: CC {msg.control} -> {param}"
                    print(self._status)
            self._pending.append(("cc", msg.control, msg.value))
        elif msg.type == "note_on" and msg.velocity > 0:
            self._last_note = msg.note
            if msg.note not in self.note_map and msg.note not in self._auto_note_order:
                self._auto_note_order.append(msg.note)
                if len(self._auto_note_order) <= len(MODE_NAMES):
                    mode = MODE_NAMES[len(self._auto_note_order) - 1]
                    self.note_map[msg.note] = mode
                    self._status = f"Auto: Note {msg.note} -> {mode}"
                    print(self._status)
            self._pending.append(("note", msg.note, msg.velocity))

    def _find_input(self) -> str | None:
        inputs = mido.get_input_names()
        return inputs[0] if inputs else None

    @staticmethod
    def list_devices() -> None:
        if mido is None:
            print("mido not installed. Run: pip install mido python-rtmidi")
            return
        inputs = mido.get_input_names()
        outputs = mido.get_output_names()
        if not inputs and not outputs:
            print("No MIDI devices found.")
            return
        print("MIDI devices:")
        for name in inputs:
            print(f"  IN:  {name}")
        for name in outputs:
            print(f"  OUT: {name}")

    def poll(self) -> list[tuple[str, int, int]]:
        events = list(self._pending)
        self._pending.clear()
        return events

    def apply_events(self, events, params, set_mode_fn) -> bool:
        """Apply MIDI events to Params and mode. Returns True if params changed."""
        from src.params import derive
        changed = False
        for etype, param, value in events:
            if etype == "cc" and param in self.cc_map:
                pname = self.cc_map[param]
                scaled = round(value / 127.0, 2)
                setattr(params, pname, scaled)
                changed = True
            elif etype == "note" and param in self.note_map:
                mode = self.note_map[param]
                set_mode_fn(mode)
                print(f"MIDI -> Mode: {mode}")
        return changed

    def learn_assign(self, param_name: str) -> bool:
        if self._last_cc is not None:
            self.cc_map = {k: v for k, v in self.cc_map.items() if v != param_name}
            self.cc_map[self._last_cc] = param_name
            self._status = f"Mapped CC {self._last_cc} -> {param_name}"
            print(self._status)
            return True
        self._status = "No CC received yet - move a knob first"
        return False

    def learn_assign_note(self, mode_name: str) -> bool:
        if self._last_note is not None:
            self.note_map = {k: v for k, v in self.note_map.items() if v != mode_name}
            self.note_map[self._last_note] = mode_name
            self._status = f"Mapped Note {self._last_note} -> {mode_name}"
            print(self._status)
            return True
        self._status = "No note received yet - press a pad first"
        return False

    def close(self) -> None:
        if self._port is not None:
            self._port.close()

    @property
    def connected(self) -> bool:
        return self._port is not None
