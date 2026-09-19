# mole_C++ Library Configuration

add_library(mole_C++
  addscalarbc.cpp
  divergence.cpp
  gradient.cpp
  interpol.cpp
  laplacian.cpp
  mixedbc.cpp
  robinbc.cpp
  utils.cpp
  interpolCtoF.cpp
  interpolCtoN.cpp
  interpolFtoC.cpp
  interpolNtoC.cpp
)
set_target_properties(
  mole_C++
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mole_C++ PUBLIC ${ARMADILLO_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIRS} ${SUPERLU_INCLUDE_DIR})
target_link_libraries(mole_C++ PUBLIC ${LINK_LIBS})

# Installation for mole library
# Install the standalone C++ library and public headers when requested.
if(MOLE_INSTALL_CPP_LIBRARY)
    install(
        TARGETS mole_C++
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION include/mole
        FILES_MATCHING
        PATTERN "*.h"
    )
endif()



# ${CMAKE_INSTALL_PREFIX}/lib for custom installation path. 
# ${CMAKE_BINARY_DIR}/install for cmake build directory installation