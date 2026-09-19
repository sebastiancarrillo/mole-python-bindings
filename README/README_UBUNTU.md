# Installing and Testing the MOLE Python Interface on Ubuntu

This guide explains how to install the MOLE Python interface directly from
source on Ubuntu. It does not require or use a prebuilt Python wheel.

The procedure has been validated on:

- Ubuntu 24.04
- Linux ARM64 (`aarch64`)
- Python 3.12
- GCC/G++
- CMake

The same procedure is intended to work on Ubuntu x86-64, although that
platform should be tested separately.

## 1. Install system dependencies

Open a terminal and install the compiler, CMake, Python, and the numerical
libraries required by MOLE:

```bash
sudo apt update
sudo apt install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  gfortran \
  libeigen3-dev \
  libopenblas-dev \
  liblapack-dev \
  python3 \
  python3-dev \
  python3-venv \
  python3-pip
```

Check that the main tools are available:

```bash
cmake --version
g++ --version
gfortran --version
python3 --version
```

## 2. Clone the repository

```bash
git clone https://github.com/sebastiancarrillo/mole-python-bindings.git
cd mole-python-bindings
```

The repository root should contain files and directories such as:

```text
CMakeLists.txt
pyproject.toml
python/
examples/
tests/
src/
```

## 3. Create a Python virtual environment

Create the environment inside the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The shell prompt should now begin with `(.venv)`. Verify that the active
Python interpreter belongs to this environment:

```bash
which python
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 4. Install the MOLE Python package from source

Install MOLE in editable mode, including the Python test dependency:

```bash
python -m pip install -e ".[test]"
```

During this operation, pip and scikit-build-core invoke CMake, compile the
original MOLE C++ implementation and its pybind11 extension, and install the
public `mole` Python package into the virtual environment.

Editable mode is useful for development: Python source changes are immediately
visible, while changes to C++ bindings or numerical C++ code still require a
rebuild.

## 5. Verify the installed package

Run the verification outside the repository so that Python cannot
accidentally import files directly from the source tree:

```bash
cd /tmp

python - <<'PY'
import platform

import mole
from scipy.sparse import csr_matrix

L = mole.laplacian_1d(
    order=2,
    cells=10,
    dx=0.1,
)

print("Architecture:", platform.machine())
print("Package:", mole.__file__)
print("Version:", mole.__version__)
print("Matrix type:", type(L))
print("Shape:", L.shape)
print("Nonzeros:", L.nnz)

assert isinstance(L, csr_matrix)
assert L.shape == (12, 12)
assert L.nnz > 0

print("PASS: the MOLE Python package works.")
PY
```

On an ARM64 machine, `platform.machine()` should report `aarch64`.

## 6. Run the installed-package tests

Return to the repository root:

```bash
cd ~/mole-python-bindings
```

If the repository was cloned elsewhere, use its actual path instead. Then run:

```bash
python -m pytest tests/wheel -v
```

Despite the directory name, these tests validate the installed public API
through `import mole`; this procedure does not build a wheel.

## 7. Run Python examples

From the repository root:

```bash
python examples/python/elliptic1d.py
```

Additional examples include:

```bash
python examples/python/transport1D.py
python examples/python/poisson_2D_dirichlet.py
python examples/python/elliptic3D.py
python examples/python/hyperbolic1D.py
```

Run the large two-dimensional benchmark with:

```bash
python examples/python/benchmark_large_2d.py \
  --cells 500 \
  --order 4 \
  --repeats 3
```

## 8. Create a complete development build

The editable installation validates the installed package. To build the C++
library, C++ examples, raw Python extension, and all internal tests together,
first make sure pybind11 is installed directly in the active environment:

```bash
python -m pip install pybind11
python -m pybind11 --cmakedir
```

Configure a development build:

```bash
cmake -S . -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOLE_BUILD_PYTHON=ON \
  -DMOLE_INSTALL_CPP_LIBRARY=ON \
  -DBUILD_TESTING=ON \
  -DMOLE_BUILD_EXAMPLES=ON \
  -DPython3_EXECUTABLE="$(which python)"
```

Build the project:

```bash
cmake --build build-linux --parallel 4
```

## 9. Run the internal Python tests

The internal tests import the freshly compiled `_mole` extension directly:

```bash
PYTHONPATH="$PWD/build-linux/python" \
python -m pytest tests/python -v
```

The validated Ubuntu ARM64 build passed all 149 internal Python tests.

## 10. Run the complete CTest suite

```bash
ctest --test-dir build-linux --output-on-failure
```

The validated configuration registered and passed:

- Five core C++ operator test executables
- The C++ scalar-boundary-condition tests
- The C++ spacing-validation tests
- The complete internal Python test suite

## 11. Build and run one C++ example

For example, compile only `elliptic1D`:

```bash
cmake --build build-linux --target elliptic1D --parallel 4
```

Run it with:

```bash
./build-linux/examples/cpp/elliptic1D
```

To build and run it in one command:

```bash
cmake --build build-linux --target elliptic1D --parallel 4 && \
./build-linux/examples/cpp/elliptic1D
```

## 12. Rebuilding after changes

After changing a C++ source file or a binding file, rebuild the development
tree:

```bash
cmake --build build-linux --parallel 4
```

Then rerun the relevant tests:

```bash
PYTHONPATH="$PWD/build-linux/python" \
python -m pytest tests/python -v
```

After changing only a Python example or Python test, recompilation is normally
not necessary.

## Troubleshooting

### `No module named pybind11`

The isolated pip build can use pybind11 without installing it permanently in
the development environment. Manual CMake configuration requires it in the
selected interpreter:

```bash
python -m pip install pybind11
```

Then repeat the CMake configuration command.

### `tests/wheel` cannot be found

Commands containing repository-relative paths must be run from the repository
root. Check the current directory:

```bash
pwd
ls
```

Then return to the repository, for example:

```bash
cd ~/mole-python-bindings
```

### Python imports an unexpected MOLE installation

Inspect the imported package:

```bash
python - <<'PY'
import mole
print(mole.__file__)
PY
```

For the editable installation, the path should resolve to the active virtual
environment and the cloned source project.

### SuperLU built-in definitions warning

The build may display:

```text
WARNING: SuperLU headers not found; using built-in definitions
```

In the validated Ubuntu ARM64 build, this message did not prevent compilation,
linking, or execution: all C++ and Python tests passed. It is a warning rather
than a test failure, but its dependency detection can be improved in future
work.

### Deactivate or reactivate the environment

Deactivate it with:

```bash
deactivate
```

Reactivate it later from the repository root with:

```bash
source .venv/bin/activate
```

## Validated result

On the tested Ubuntu ARM64 virtual machine:

```text
149 internal Python tests passed
100% of registered CTest tests passed
0 CTest failures
```

This confirms that the MOLE C++ library and its Python interface can be built
and tested directly from source on Ubuntu ARM64 without using a prebuilt wheel.
