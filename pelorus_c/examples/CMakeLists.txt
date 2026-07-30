cmake_minimum_required(VERSION 3.12)
project(pelorus_examples LANGUAGES C CXX)

# Pelorus build type (used to locate the Cargo-built shared library).
if(NOT DEFINED PELORUS_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PELORUS_BUILD_TYPE_DEFAULT "debug")
    else()
        set(PELORUS_BUILD_TYPE_DEFAULT "release")
    endif()
    set(PELORUS_BUILD_TYPE "${PELORUS_BUILD_TYPE_DEFAULT}" CACHE STRING
        "Pelorus Cargo build type (debug or release)")
endif()
set_property(CACHE PELORUS_BUILD_TYPE PROPERTY STRINGS debug release)
string(TOLOWER "${PELORUS_BUILD_TYPE}" PELORUS_BUILD_TYPE_LOWER)
set(PELORUS_TARGET_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../target/${PELORUS_BUILD_TYPE_LOWER}")
message(STATUS "Pelorus target dir: ${PELORUS_TARGET_DIR}")

find_package(Eigen3 QUIET)

if(NOT DEFINED PELORUS_EXAMPLE_LIBRARY)
    message(FATAL_ERROR
        "Configure examples through pelorus_c with PELORUS_BUILD_EXAMPLES=ON")
endif()

# C examples
add_executable(async_callback async_callback.c)
target_link_libraries(async_callback ${PELORUS_EXAMPLE_LIBRARY} m)
set_target_properties(async_callback PROPERTIES
    BUILD_RPATH "${PELORUS_TARGET_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
)

add_executable(sync_blocking sync_blocking.c)
target_link_libraries(sync_blocking ${PELORUS_EXAMPLE_LIBRARY} m)
set_target_properties(sync_blocking PROPERTIES
    BUILD_RPATH "${PELORUS_TARGET_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
)

# C++ PCL example (PCL 1.15 requires C++17 or later)
find_package(PCL COMPONENTS common)
if(PCL_FOUND)
    message(STATUS "PCL found - building C++ API example")
    add_executable(cpp_wrapper sync_blocking_pcl_eigen.cpp)
    target_link_libraries(cpp_wrapper ${PELORUS_EXAMPLE_LIBRARY} ${PCL_LIBRARIES})
    target_include_directories(cpp_wrapper PRIVATE ${PCL_INCLUDE_DIRS})
    if(PCL_DEFINITIONS)
        separate_arguments(PCL_DEFINITIONS)
        target_compile_options(cpp_wrapper PRIVATE ${PCL_DEFINITIONS})
    endif()
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(cpp_wrapper Eigen3::Eigen)
    endif()
    set_target_properties(cpp_wrapper PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH "${PELORUS_TARGET_DIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
    )
    set(CPP_WRAPPER_TARGET cpp_wrapper)
else()
    message(STATUS "PCL not found - skipping C++ API example")
    set(CPP_WRAPPER_TARGET "")
endif()

# Eigen3 example (optional, requires Eigen3)
if(Eigen3_FOUND OR EIGEN3_FOUND)
    message(STATUS "Eigen3 found - building eigen_example")
    add_executable(eigen_example eigen_example.cpp)
    target_link_libraries(eigen_example ${PELORUS_EXAMPLE_LIBRARY} Eigen3::Eigen)
    set_target_properties(eigen_example PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH "${PELORUS_TARGET_DIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
    )
    set(EIGEN_EXAMPLE_TARGET eigen_example)
else()
    message(STATUS "Eigen3 not found - skipping eigen_example")
    set(EIGEN_EXAMPLE_TARGET "")
endif()

# Install examples (optional)
install(TARGETS async_callback sync_blocking ${CPP_WRAPPER_TARGET} ${EIGEN_EXAMPLE_TARGET}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/pelorus_examples
)
