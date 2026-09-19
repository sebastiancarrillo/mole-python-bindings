"""Tests for the P and Q weights exposed by MOLE."""

import numpy as np
import pytest

import _mole


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_gradient_weights_1d(order):
    """Test the P weights of the non-periodic 1D gradient."""

    cells = 2 * order + 1
    dx = 0.25

    weights = _mole.gradient_weights_1d(
        order,
        cells,
        dx,
    )

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert weights.ndim == 1
    assert weights.size > 0
    assert np.all(np.isfinite(weights))

    # Calling the same C++ constructor twice must produce identical weights
    repeated_weights = _mole.gradient_weights_1d(
        order,
        cells,
        dx,
    )

    np.testing.assert_array_equal(
        weights,
        repeated_weights,
    )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_divergence_weights_1d(order):
    """Test the Q weights of the non-periodic 1D divergence."""

    cells = 2 * order + 1
    dx = 0.25

    weights = _mole.divergence_weights_1d(
        order,
        cells,
        dx,
    )

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert weights.ndim == 1
    assert weights.size > 0
    assert np.all(np.isfinite(weights))

    repeated_weights = _mole.divergence_weights_1d(
        order,
        cells,
        dx,
    )

    np.testing.assert_array_equal(
        weights,
        repeated_weights,
    )


@pytest.mark.parametrize(
    "function",
    [
        _mole.gradient_weights_1d,
        _mole.divergence_weights_1d,
    ],
)
@pytest.mark.parametrize(
    "invalid_dx",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
    ],
)
def test_weights_reject_invalid_spacing(function, invalid_dx):
    with pytest.raises(ValueError):
        function(
            2,
            10,
            invalid_dx,
        )


@pytest.mark.parametrize(
    "function_name",
    [
        "gradient_weights_1d",
        "divergence_weights_1d",
    ],
)
def test_weight_function_is_exposed(function_name):
    assert hasattr(_mole, function_name)
    assert callable(getattr(_mole, function_name))