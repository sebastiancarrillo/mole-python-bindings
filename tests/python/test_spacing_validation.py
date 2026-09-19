"""Spacing validation tests translated from the C++ suite."""

import math

import pytest

import _mole


@pytest.mark.parametrize(
    "spacing",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_gradient_1d_rejects_invalid_dx(spacing):
    with pytest.raises(ValueError):
        _mole.gradient_1d(2, 10, spacing)


def test_gradient_2d_rejects_zero_dy():
    with pytest.raises(ValueError):
        _mole.gradient_2d(
            2,
            10,
            10,
            0.1,
            0.0,
        )


def test_gradient_3d_rejects_negative_dz():
    with pytest.raises(ValueError):
        _mole.gradient_3d(
            2,
            10,
            10,
            10,
            0.1,
            0.1,
            -0.1,
        )


def test_divergence_1d_rejects_zero_dx():
    with pytest.raises(ValueError):
        _mole.divergence_1d(
            2,
            10,
            0.0,
        )


def test_divergence_3d_rejects_infinite_dz():
    with pytest.raises(ValueError):
        _mole.divergence_3d(
            2,
            10,
            10,
            10,
            0.1,
            0.1,
            math.inf,
        )


def test_laplacian_3d_rejects_zero_dz():
    with pytest.raises(ValueError):
        _mole.laplacian_3d(
            2,
            10,
            10,
            10,
            0.1,
            0.1,
            0.0,
        )


def test_robin_bc_rejects_zero_dx():
    with pytest.raises(ValueError):
        _mole.robin_bc_1d(
            2,
            10,
            0.0,
            1.0,
            1.0,
        )


def test_mixed_bc_rejects_negative_dx():
    with pytest.raises(ValueError):
        _mole.mixed_bc_1d(
            2,
            10,
            -0.1,
            "Dirichlet",
            [1.0],
            "Dirichlet",
            [1.0],
        )


def test_gradient_accepts_valid_spacing():
    operator = _mole.gradient_1d(
        2,
        10,
        0.1,
    )

    assert operator.shape == (11, 12)


def test_laplacian_3d_accepts_valid_spacing():
    operator = _mole.laplacian_3d(
        2,
        10,
        10,
        10,
        0.1,
        0.1,
        0.1,
    )

    assert operator.shape == (12**3, 12**3)


def test_error_message_names_dx():
    with pytest.raises(ValueError, match="dx"):
        _mole.gradient_1d(
            2,
            10,
            0.0,
        )