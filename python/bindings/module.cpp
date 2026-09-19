/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Main Python module definition for MOLE.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;


// Registration functions implemented in the other binding files.
void bind_operators(py::module_& module);
void bind_interpolation(py::module_& module);
void bind_boundary_conditions(py::module_& module);
void bind_add_scalar_bc(py::module_& module);


PYBIND11_MODULE(_mole, module)
{
    module.doc() =
        "Python bindings for the Mimetic Operators Library Enhanced";

    module.attr("__version__") = "0.1.0.dev0";

    bind_operators(module);
    bind_interpolation(module);
    bind_boundary_conditions(module);
    bind_add_scalar_bc(module);
}