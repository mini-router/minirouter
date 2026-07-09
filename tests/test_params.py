"""Offline unit tests for θ packing/unpacking (SPEC smoke test S3).

``coordinator/params.py`` defines the flat CMA-ES search vector layout:
``θ = [head_W.flatten() | svf_scales]``. The smoke ladder exercises this in
``tests/smoke/run_smoke.py::s3``, but there was no dedicated pytest module to
lock round-trip integrity, dtype, and validation offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.coordinator import params as P


def test_make_spec_defaults_match_trinity_layout():
    spec = P.make_spec()
    assert spec.head_shape == (6, 1024)
    assert spec.n_head == 6144
    assert spec.n_svf == 7168
    assert spec.n_total == 13312


def test_pack_unpack_round_trip_random():
    spec = P.make_spec()
    rng = np.random.default_rng(42)
    head_W = rng.standard_normal(spec.head_shape)
    svf_scales = rng.standard_normal(spec.n_svf)
    theta = P.pack(head_W, svf_scales)
    head2, svf2 = P.unpack(theta, spec)
    assert np.allclose(head_W, head2)
    assert np.allclose(svf_scales, svf2)


def test_pack_returns_float64_contiguous_vector():
    spec = P.make_spec()
    head_W = np.ones(spec.head_shape, dtype=np.float32)
    svf_scales = np.arange(spec.n_svf, dtype=np.int32)
    theta = P.pack(head_W, svf_scales)
    assert theta.dtype == np.float64
    assert theta.flags["C_CONTIGUOUS"]
    assert theta.shape == (spec.n_total,)


def test_unpack_returns_copies_not_views():
    spec = P.make_spec()
    theta = P.initial_theta(spec)
    head_W, svf_scales = P.unpack(theta, spec)
    head_W[0, 0] = 99.0
    svf_scales[0] = 99.0
    head3, svf3 = P.unpack(theta, spec)
    assert head3[0, 0] == 0.0
    assert svf3[0] == 1.0


def test_initial_theta_is_zero_head_and_unit_svf_scales():
    spec = P.make_spec()
    theta = P.initial_theta(spec)
    head_W, svf_scales = P.unpack(theta, spec)
    assert np.allclose(head_W, 0.0)
    assert np.allclose(svf_scales, 1.0)


def test_pack_preserves_row_major_head_layout():
    spec = P.make_spec(n_a=2, d_h=3, n_svf=4)
    head_W = np.arange(6, dtype=np.float64).reshape(2, 3)
    svf_scales = np.array([10.0, 11.0, 12.0, 13.0])
    theta = P.pack(head_W, svf_scales)
    assert list(theta[:6]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(theta[6:]) == [10.0, 11.0, 12.0, 13.0]


@pytest.mark.parametrize(
    ("head_shape", "svf_len"),
    [
        ((6, 1024), 7168),
        ((4, 8), 16),
    ],
)
def test_round_trip_across_custom_specs(head_shape, svf_len):
    n_a, d_h = head_shape
    spec = P.make_spec(n_a=n_a, d_h=d_h, n_svf=svf_len)
    head_W = np.linspace(0, 1, spec.n_head, dtype=np.float64).reshape(head_shape)
    svf_scales = np.linspace(2, 3, spec.n_svf, dtype=np.float64)
    theta = P.pack(head_W, svf_scales)
    assert theta.size == spec.n_total
    head2, svf2 = P.unpack(theta, spec)
    assert np.array_equal(head_W, head2)
    assert np.array_equal(svf_scales, svf2)


def test_pack_rejects_non_2d_head():
    with pytest.raises(ValueError, match="head_W must be 2-D"):
        P.pack(np.zeros(10), np.zeros(5))


def test_pack_rejects_non_1d_svf_scales():
    with pytest.raises(ValueError, match="svf_scales must be 1-D"):
        P.pack(np.zeros((2, 3)), np.zeros((2, 3)))


def test_unpack_rejects_wrong_length_theta():
    spec = P.make_spec()
    with pytest.raises(ValueError, match="expected n_total"):
        P.unpack(np.zeros(spec.n_total + 1), spec)


def test_param_spec_rejects_inconsistent_n_head():
    with pytest.raises(ValueError, match="n_head"):
        P.ParamSpec(head_shape=(6, 1024), n_head=100, n_svf=7168, n_total=7268)


def test_param_spec_rejects_inconsistent_n_total():
    with pytest.raises(ValueError, match="n_total"):
        P.ParamSpec(head_shape=(6, 1024), n_head=6144, n_svf=7168, n_total=9999)
