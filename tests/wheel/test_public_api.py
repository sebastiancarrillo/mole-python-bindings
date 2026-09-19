"""Integration tests for the installed public MOLE Python package."""

from importlib.metadata import version

import mole
import pytest
from scipy.sparse import csr_matrix


EXPECTED_PUBLIC_API = {
    "add_scalar_bc_1d",
    "add_scalar_bc_2d",
    "add_scalar_bc_3d",
    "divergence_1d",
    "divergence_1d_bc",
    "divergence_2d",
    "divergence_2d_bc",
    "divergence_3d",
    "divergence_3d_bc",
    "divergence_weights_1d",
    "gradient_1d",
    "gradient_1d_bc",
    "gradient_2d",
    "gradient_2d_bc",
    "gradient_3d",
    "gradient_3d_bc",
    "gradient_weights_1d",
    "interpolate_cell_to_face_1d",
    "interpolate_cell_to_face_2d",
    "interpolate_cell_to_face_3d",
    "interpolate_cell_to_node_1d",
    "interpolate_cell_to_node_2d",
    "interpolate_cell_to_node_3d",
    "interpolate_face_to_cell_1d",
    "interpolate_face_to_cell_2d",
    "interpolate_face_to_cell_3d",
    "interpolate_node_to_cell_1d",
    "interpolate_node_to_cell_2d",
    "interpolate_node_to_cell_3d",
    "interpolation_1d",
    "interpolation_2d",
    "interpolation_3d",
    "interpolation_alternate_1d",
    "interpolation_alternate_2d",
    "interpolation_alternate_3d",
    "laplacian_1d",
    "laplacian_2d",
    "laplacian_3d",
    "mixed_bc_1d",
    "mixed_bc_2d",
    "mixed_bc_3d",
    "robin_bc_1d",
    "robin_bc_2d",
    "robin_bc_3d",
}


def test_distribution_and_package_versions_match():
    """The distribution and public module must report the same version."""

    assert version("mole-mimetic") == "0.1.0"
    assert mole.__version__ == "0.1.0"


def test_public_api_is_complete():
    """The installed package must expose the complete documented API."""

    assert set(mole.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.parametrize("function_name", sorted(EXPECTED_PUBLIC_API))
def test_public_functions_are_callable(function_name):
    """Every public API entry must exist and be callable."""

    assert hasattr(mole, function_name)

    function = getattr(mole, function_name)

    assert callable(function)


def test_internal_extension_is_not_public():
    """The compiled implementation module must not be part of __all__."""

    assert "_mole" not in mole.__all__


def test_public_operator_returns_csr_matrix():
    """Public operator constructors must return SciPy CSR matrices."""

    operator = mole.laplacian_1d(
        order=2,
        cells=10,
        dx=0.1,
    )

    assert isinstance(operator, csr_matrix)
    assert operator.shape == (12, 12)
    assert operator.nnz > 0


def test_operator_can_be_applied():
    """An operator obtained through the public API must be usable."""

    import numpy as np

    operator = mole.laplacian_1d(2, 10, 0.1)
    constant = np.ones(operator.shape[1])
    result = operator @ constant

    assert result.shape == (operator.shape[0],)
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) < 1e-10