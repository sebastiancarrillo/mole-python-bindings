# Python interface for MOLE

find_package(
    Python3
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

# Locate pybind11 from the currently active Python environment.
execute_process(
    COMMAND
        "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE
        pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _mole
    MODULE
    bindings/module.cpp
    bindings/operators.cpp
    bindings/interpolation.cpp
    bindings/boundary_conditions.cpp
    bindings/add_scalar_bc.cpp
)

target_include_directories(
    _mole
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src/cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
)

target_link_libraries(
    _mole
    PRIVATE
    mole_C++
)

target_compile_features(
    _mole
    PRIVATE
    cxx_std_14
)

set_target_properties(
    _mole
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python"
)

if(APPLE)
    set_target_properties(
        _mole
        PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH OFF
    )
elseif(UNIX)
    set_target_properties(
        _mole
        PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH OFF
    )
endif()

install(
    TARGETS _mole
    LIBRARY DESTINATION mole
    RUNTIME DESTINATION mole
)

# Armadillo is a runtime dependency of _mole.
if(APPLE)
    install(
        FILES
            "${ARMADILLO_INSTALL_DIR}/lib/libarmadillo.${ARMADILLO_VERSION}.dylib"
            "${ARMADILLO_INSTALL_DIR}/lib/libarmadillo.15.dylib"
        DESTINATION mole
    )
endif()