"""
Python bindings for the Mimetic Operators Library Enhanced
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing

from scipy.sparse import csr_matrix

__all__: list[str] = ['add_scalar_bc_1d', 'add_scalar_bc_2d', 'add_scalar_bc_3d', 'divergence_1d', 'divergence_1d_bc', 'divergence_2d', 'divergence_2d_bc', 'divergence_3d', 'divergence_3d_bc', 'divergence_weights_1d', 'gradient_1d', 'gradient_1d_bc', 'gradient_2d', 'gradient_2d_bc', 'gradient_3d', 'gradient_3d_bc', 'gradient_weights_1d', 'interpolate_cell_to_face_1d', 'interpolate_cell_to_face_2d', 'interpolate_cell_to_face_3d', 'interpolate_cell_to_node_1d', 'interpolate_cell_to_node_2d', 'interpolate_cell_to_node_3d', 'interpolate_face_to_cell_1d', 'interpolate_face_to_cell_2d', 'interpolate_face_to_cell_3d', 'interpolate_node_to_cell_1d', 'interpolate_node_to_cell_2d', 'interpolate_node_to_cell_3d', 'interpolation_1d', 'interpolation_2d', 'interpolation_3d', 'interpolation_alternate_1d', 'interpolation_alternate_2d', 'interpolation_alternate_3d', 'laplacian_1d', 'laplacian_2d', 'laplacian_3d', 'mixed_bc_1d', 'mixed_bc_2d', 'mixed_bc_3d', 'robin_bc_1d', 'robin_bc_2d', 'robin_bc_3d']
def add_scalar_bc_1d(matrix: csr_matrix, rhs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> tuple[csr_matrix, numpy.typing.NDArray[numpy.float64]]:
    """
    Apply scalar boundary conditions to a 1-D linear system.
    
    Parameters
    ----------
    matrix
        Square SciPy sparse matrix.
    rhs
        One-dimensional NumPy right-hand side.
    order
        Mimetic accuracy order.
    cells
        Number of cells.
    dx
        Cell spacing.
    dirichlet
        Two coefficients ordered as [left, right].
    neumann
        Two coefficients ordered as [left, right].
    values
        Two boundary values ordered as [left, right].
    
    Returns
    -------
    matrix, rhs
        Modified SciPy CSR matrix and NumPy array.
    """
def add_scalar_bc_2d(matrix: csr_matrix, rhs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], values: collections.abc.Sequence) -> tuple[csr_matrix, numpy.typing.NDArray[numpy.float64]]:
    """
    Apply scalar boundary conditions to a 2-D linear system.
    
    Boundary order:
    
        left, right, bottom, top
    
    `values` must be a sequence containing four one-dimensional arrays.
    
    Returns the modified SciPy CSR matrix and NumPy RHS.
    """
def add_scalar_bc_3d(matrix: csr_matrix, rhs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], values: collections.abc.Sequence) -> tuple[csr_matrix, numpy.typing.NDArray[numpy.float64]]:
    """
    Apply scalar boundary conditions to a 3-D linear system.
    
    Boundary order:
    
        left, right, bottom, top, front, back
    
    `values` must be a sequence containing six one-dimensional arrays.
    
    Returns the modified SciPy CSR matrix and NumPy RHS.
    """
def divergence_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 1-D mimetic divergence.
    """
def divergence_1d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 1-D divergence.
    
    All-zero Dirichlet and Neumann arrays create a periodic operator.
    The arrays must contain [left, right].
    """
def divergence_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 2-D mimetic divergence.
    """
def divergence_2d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 2-D divergence.
    
    Boundary order: left, right, bottom, top.
    An axis is periodic when its corresponding coefficients are zero.
    """
def divergence_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 3-D mimetic divergence.
    """
def divergence_3d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 3-D divergence.
    
    Boundary order: left, right, bottom, top, front, back.
    An axis is periodic when its corresponding coefficients are zero.
    """
def divergence_weights_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Return the Q weights of a non-periodic 1-D divergence.
    """
def gradient_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 1-D mimetic gradient.
    """
def gradient_1d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 1-D gradient.
    
    All-zero Dirichlet and Neumann arrays create a periodic operator.
    The arrays must contain [left, right].
    """
def gradient_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 2-D mimetic gradient.
    """
def gradient_2d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 2-D gradient.
    
    Boundary order: left, right, bottom, top.
    An axis is periodic when its corresponding coefficients are zero.
    """
def gradient_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a non-periodic 3-D mimetic gradient.
    """
def gradient_3d_bc(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    """
    Construct a boundary-aware 3-D gradient.
    
    Boundary order: left, right, bottom, top, front, back.
    An axis is periodic when its corresponding coefficients are zero.
    """
def gradient_weights_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Return the P weights of a non-periodic 1-D gradient.
    """
def interpolate_cell_to_face_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_cell_to_face_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_cell_to_face_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_cell_to_node_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_cell_to_node_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_cell_to_node_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_face_to_cell_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_face_to_cell_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_face_to_cell_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_node_to_cell_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_node_to_cell_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolate_node_to_cell_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dirichlet: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], neumann: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> csr_matrix:
    ...
def interpolation_1d(cells: typing.SupportsInt | typing.SupportsIndex, weight: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 1-D mimetic interpolation operator.
    """
def interpolation_2d(cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, weight_x: typing.SupportsFloat | typing.SupportsIndex, weight_y: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 2-D mimetic interpolation operator.
    """
def interpolation_3d(cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, weight_x: typing.SupportsFloat | typing.SupportsIndex, weight_y: typing.SupportsFloat | typing.SupportsIndex, weight_z: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 3-D mimetic interpolation operator.
    """
def interpolation_alternate_1d(type: bool, cells: typing.SupportsInt | typing.SupportsIndex, weight: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct the alternate 1-D interpolation operator.
    """
def interpolation_alternate_2d(type: bool, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, weight_x: typing.SupportsFloat | typing.SupportsIndex, weight_y: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct the alternate 2-D interpolation operator.
    """
def interpolation_alternate_3d(type: bool, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, weight_x: typing.SupportsFloat | typing.SupportsIndex, weight_y: typing.SupportsFloat | typing.SupportsIndex, weight_z: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct the alternate 3-D interpolation operator.
    """
def laplacian_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 1-D mimetic Laplacian.
    """
def laplacian_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 2-D mimetic Laplacian.
    """
def laplacian_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 3-D mimetic Laplacian.
    """
def mixed_bc_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, left_type: str, left_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], right_type: str, right_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> csr_matrix:
    """
    Construct a 1-D mixed boundary-condition operator.
    
    Valid boundary types are:
    
        "Dirichlet"
        "Neumann"
        "Robin"
    """
def mixed_bc_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, left_type: str, left_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], right_type: str, right_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], bottom_type: str, bottom_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], top_type: str, top_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> csr_matrix:
    """
    Construct a 2-D mixed boundary-condition operator.
    
    Boundary order:
    
        left, right, bottom, top
    """
def mixed_bc_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex, left_type: str, left_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], right_type: str, right_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], bottom_type: str, bottom_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], top_type: str, top_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], front_type: str, front_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], back_type: str, back_coefficients: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> csr_matrix:
    """
    Construct a 3-D mixed boundary-condition operator.
    
    Boundary order:
    
        left, right, bottom, top, front, back
    """
def robin_bc_1d(order: typing.SupportsInt | typing.SupportsIndex, cells: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dirichlet_coefficient: typing.SupportsFloat | typing.SupportsIndex, neumann_coefficient: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 1-D Robin boundary-condition operator.
    
    The boundary equation has the form:
    
        a*u + b*du/dn = g
    
    where `dirichlet_coefficient` is a and
    `neumann_coefficient` is b.
    """
def robin_bc_2d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dirichlet_coefficient: typing.SupportsFloat | typing.SupportsIndex, neumann_coefficient: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 2-D Robin boundary-condition operator.
    """
def robin_bc_3d(order: typing.SupportsInt | typing.SupportsIndex, cells_x: typing.SupportsInt | typing.SupportsIndex, cells_y: typing.SupportsInt | typing.SupportsIndex, cells_z: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, dy: typing.SupportsFloat | typing.SupportsIndex, dz: typing.SupportsFloat | typing.SupportsIndex, dirichlet_coefficient: typing.SupportsFloat | typing.SupportsIndex, neumann_coefficient: typing.SupportsFloat | typing.SupportsIndex) -> csr_matrix:
    """
    Construct a 3-D Robin boundary-condition operator.
    """
__version__: str = '0.1.0'