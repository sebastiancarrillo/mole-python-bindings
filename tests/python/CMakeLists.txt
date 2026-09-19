# Python test suite

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Confirm that pytest is installed in the selected Python environment.
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c "import pytest"
  RESULT_VARIABLE PYTEST_IMPORT_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)

if(NOT PYTEST_IMPORT_RESULT EQUAL 0)
  message(
    FATAL_ERROR
    "pytest is not installed for ${Python3_EXECUTABLE}. "
    "Install it with: ${Python3_EXECUTABLE} -m pip install pytest"
  )
endif()

add_test(
  NAME python_tests
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=$<TARGET_FILE_DIR:_mole>"
    "${Python3_EXECUTABLE}" -m pytest
    "${CMAKE_CURRENT_SOURCE_DIR}"
    -q
)

set_tests_properties(
  python_tests
  PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Ensure that the Python extension exists before run_tests invokes CTest.
if(TARGET run_tests)
  add_dependencies(run_tests _mole)
endif()