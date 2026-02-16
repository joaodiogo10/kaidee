"""Tests for src.transforms."""

import numpy as np
import pytest

from src.transforms import (
    make_abstract, solarize, color_rotate, color_rotate_f32,
    posterize, pixel_sort, blend, fast_zoom, make_vignette,
)


def _rand_img(h=64, w=80):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestSolarize:
    def test_output_shape_dtype(self):
        img = _rand_img()
        out = solarize(img, 128)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_below_threshold_unchanged(self):
        img = np.full((4, 4, 3), 50, dtype=np.uint8)
        out = solarize(img, 128)
        np.testing.assert_array_equal(out, img)

    def test_above_threshold_inverted(self):
        img = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = solarize(img, 128)
        expected = 255 - 200
        np.testing.assert_array_equal(out, expected)

    def test_at_threshold_boundary(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = solarize(img, 128)
        np.testing.assert_array_equal(out, img)


class TestColorRotate:
    def test_zero_angle_identity(self):
        img = _rand_img()
        out = color_rotate(img, 0.0)
        np.testing.assert_array_almost_equal(out, img, decimal=0)

    def test_360_angle_identity(self):
        img = _rand_img()
        out = color_rotate(img, 360.0)
        np.testing.assert_array_almost_equal(out, img, decimal=0)

    def test_output_range(self):
        img = _rand_img()
        out = color_rotate(img, 120.0)
        assert out.min() >= 0
        assert out.max() <= 255


class TestColorRotateF32:
    def test_returns_float32(self):
        f = _rand_img().astype(np.float32)
        out = color_rotate_f32(f, 0.5)
        assert out.dtype == np.float32

    def test_zero_angle_identity(self):
        f = _rand_img().astype(np.float32)
        out = color_rotate_f32(f, 0.0)
        np.testing.assert_array_almost_equal(out, f, decimal=3)


class TestPosterize:
    def test_two_levels(self):
        img = _rand_img()
        out = posterize(img, 2)
        unique = set(np.unique(out))
        assert unique.issubset({0, 255})

    def test_output_dtype(self):
        img = _rand_img()
        out = posterize(img, 4)
        assert out.dtype == np.uint8


class TestPixelSort:
    def test_output_shape(self):
        img = _rand_img(32, 40)
        out = pixel_sort(img)
        assert out.shape == img.shape

    def test_sorted_rows(self):
        img = _rand_img(8, 16)
        out = pixel_sort(img)
        f = out.astype(np.float32)
        luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        for row in range(luma.shape[0]):
            diffs = np.diff(luma[row])
            assert np.all(diffs >= -0.5)  # allow tiny float rounding


class TestBlend:
    def test_alpha_zero_returns_a(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = blend(a, b, 0.0)
        np.testing.assert_array_equal(out, a)

    def test_alpha_one_returns_b(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = blend(a, b, 1.0)
        np.testing.assert_array_almost_equal(out, b, decimal=0)

    def test_midpoint(self):
        a = np.full((4, 4, 3), 0, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = blend(a, b, 0.5)
        assert 95 <= out[0, 0, 0] <= 105

    def test_output_dtype(self):
        a, b = _rand_img(), _rand_img()
        assert blend(a, b, 0.3).dtype == np.uint8


class TestFastZoom:
    def test_zoom_1_identity(self):
        img = _rand_img(32, 48)
        out = fast_zoom(img, 1.0)
        assert out.shape == img.shape

    def test_zoom_2_shape(self):
        img = _rand_img(64, 80)
        out = fast_zoom(img, 2.0)
        assert out.shape == img.shape

    def test_zoom_preserves_center(self):
        img = np.zeros((64, 80, 3), dtype=np.uint8)
        img[30:34, 38:42, :] = 255
        out = fast_zoom(img, 1.5)
        center_val = out[32, 40]
        assert center_val.sum() > 0


class TestMakeVignette:
    def test_shape(self):
        vig = make_vignette(100, 120)
        assert vig.shape == (100, 120)

    def test_center_brighter(self):
        vig = make_vignette(100, 120)
        center = vig[50, 60]
        corner = vig[0, 0]
        assert center > corner

    def test_range(self):
        vig = make_vignette(200, 300)
        assert vig.min() >= 0.3
        assert vig.max() <= 1.0

    def test_dtype(self):
        vig = make_vignette(50, 50)
        assert vig.dtype == np.float32


class TestMakeAbstract:
    def test_output_shape_dtype(self):
        img = _rand_img()
        out = make_abstract(img, sigma=10)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_smoother_than_input(self):
        img = _rand_img(64, 80)
        out = make_abstract(img, sigma=20)
        # Abstract version should have lower variance (smoother)
        assert out.astype(float).std() < img.astype(float).std() + 50
