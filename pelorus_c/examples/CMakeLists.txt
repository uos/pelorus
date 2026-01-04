cmake_minimum_required(VERSION 3.12)
project(pelorus_examples LANGUAGES C CXX)

# Pelorus build-type option (used to locate built shared library for RPATH)
option(PELORUS_BUILD_TYPE "Pelorus build type (debug or release)" "release")
string(TOLOWER ${PELORUS_BUILD_TYPE} PELORUS_BUILD_TYPE_LOWER)
set(PELORUS_TARGET_DIR "${CMAKE_SOURCE_DIR}/../target/${PELORUS_BUILD_TYPE_LOWER}")
message(STATUS "Pelorus target dir: ${PELORUS_TARGET_DIR}")

# C examples
add_executable(async_callback async_callback.c)
target_link_libraries(async_callback pelorus m)
set_target_properties(async_callback PROPERTIES
    BUILD_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
)

add_executable(sync_blocking sync_blocking.c)
target_link_libraries(sync_blocking pelorus m)
set_target_properties(sync_blocking PROPERTIES
    BUILD_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
)

# C++ example (requires C++11 or later)
find_package(PCL)
if(PCL_FOUND)
    message(STATUS "PCL found - building C++ API example")
    add_executable(cpp_wrapper sync_blocking_pcl_eigen.cpp)
    target_link_libraries(cpp_wrapper pelorus ${PCL_LIBRARIES})
    target_include_directories(cpp_wrapper PRIVATE ${PCL_INCLUDE_DIRS})
    if(PCL_DEFINITIONS)
        separate_arguments(PCL_DEFINITIONS)
        target_compile_options(cpp_wrapper PRIVATE ${PCL_DEFINITIONS})
    endif()
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(cpp_wrapper Eigen3::Eigen)
    endif()
    set_target_properties(cpp_wrapper PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
    )
    set(CPP_WRAPPER_TARGET cpp_wrapper)
else()
    message(STATUS "PCL not found - skipping C++ API example")
    set(CPP_WRAPPER_TARGET "")
endif()

# Eigen3 example (optional, requires Eigen3)

find_package(Eigen3 QUIET)
if(Eigen3_FOUND OR EIGEN3_FOUND)
    message(STATUS "Eigen3 found - building eigen_example")
    add_executable(eigen_example eigen_example.cpp)
    target_link_libraries(eigen_example pelorus Eigen3::Eigen)
    set_target_properties(eigen_example PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../../../target/${PELORUS_BUILD_TYPE_LOWER}"
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
