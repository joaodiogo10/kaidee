"""Tests for src.params."""

import json
import tempfile
import os

import pytest

from src.params import Params, DerivedParams, Preset, derive, load_presets, save_presets


class TestParams:
    def test_defaults(self):
        p = Params()
        assert p.reactivity == 0.5
        assert p.brightness == 0.5

    def test_to_dict(self):
        p = Params(reactivity=0.3)
        d = p.to_dict()
        assert d["reactivity"] == 0.3
        assert "brightness" in d

    def test_from_dict(self):
        p = Params.from_dict({"reactivity": 0.8, "movement": 0.1})
        assert p.reactivity == 0.8
        assert p.movement == 0.1
        assert p.color == 0.5  # default

    def test_from_dict_ignores_unknown(self):
        p = Params.from_dict({"reactivity": 0.5, "unknown_field": 99})
        assert p.reactivity == 0.5

    def test_adjust_clamps(self):
        p = Params(reactivity=0.95)
        p.adjust("reactivity", 0.1)
        assert p.reactivity == 1.0

        p.adjust("reactivity", -2.0)
        assert p.reactivity == 0.0

    def test_adjust_normal(self):
        p = Params(movement=0.5)
        p.adjust("movement", 0.05)
        assert abs(p.movement - 0.55) < 0.001

    def test_roundtrip(self):
        p = Params(reactivity=0.1, perception=0.9, movement=0.3,
                   color=0.7, brightness=0.4, blend=0.6, img_blend=0.2)
        d = p.to_dict()
        p2 = Params.from_dict(d)
        assert p == p2


class TestDerive:
    def test_returns_derived_params(self):
        dp = derive(Params())
        assert isinstance(dp, DerivedParams)

    def test_zero_reactivity(self):
        dp = derive(Params(reactivity=0.0))
        assert dp.react_mod == pytest.approx(0.2)

    def test_max_reactivity(self):
        dp = derive(Params(reactivity=1.0))
        assert dp.react_mod == pytest.approx(3.0)

    def test_code_param(self):
        dp = derive(Params(code=0.0))
        assert dp.code_vis == 0.0
        assert dp.code_blend == 0.0
        dp = derive(Params(code=1.0))
        assert dp.code_vis == pytest.approx(2.0)
        assert dp.code_blend == pytest.approx(1.0)

    def test_movement_affects_rotation(self):
        dp_low = derive(Params(movement=0.0))
        dp_high = derive(Params(movement=1.0))
        assert dp_high.rotation_speed > dp_low.rotation_speed

    def test_img_cycle_bars_discrete(self):
        dp = derive(Params(perception=0.0))
        assert dp.img_cycle_bars == 16
        dp = derive(Params(perception=1.0))
        assert dp.img_cycle_bars == 2

    def test_chaos_manual_disabled_at_low_perception(self):
        dp = derive(Params(perception=0.0))
        assert dp.chaos_manual == -1.0

    def test_img_layers(self):
        dp = derive(Params(img_blend=0.0))
        assert dp.img_layers == 1
        dp = derive(Params(img_blend=1.0))
        assert dp.img_layers >= 3


class TestPreset:
    def test_roundtrip(self):
        p = Preset(params=Params(reactivity=0.8), mode="glitch")
        d = p.to_dict()
        p2 = Preset.from_dict(d)
        assert p2.mode == "glitch"
        assert p2.params.reactivity == 0.8

    def test_default_mode(self):
        p = Preset.from_dict({"reactivity": 0.5})
        assert p.mode == "auto"


class TestPresetIO:
    def test_save_load_roundtrip(self):
        presets = {
            "1": Preset(params=Params(reactivity=0.3), mode="feedback"),
            "2": Preset(params=Params(color=0.9), mode="strobe"),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_presets(presets, path)
            loaded = load_presets(path)
            assert "1" in loaded
            assert loaded["1"].mode == "feedback"
            assert loaded["1"].params.reactivity == 0.3
            assert loaded["2"].params.color == 0.9
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        result = load_presets("/nonexistent/path.json")
        assert result == {}

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json {{{")
            path = f.name
        try:
            result = load_presets(path)
            assert result == {}
        finally:
            os.unlink(path)
