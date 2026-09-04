cmake_minimum_required(VERSION 3.16)
project(pelorus_glim_odometry LANGUAGES CXX)

find_package(glim REQUIRED)
find_package(GTSAM 4.2 REQUIRED)
find_package(gtsam_points 1.1.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default install prefix if not specified. Only when this is the top-level
# project: as a subdirectory of the ament package the parent owns the prefix
# (colcon passes its own), and a subproject must not force it.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
   AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "Default install prefix" FORCE)
endif()

# Try to find installed pelorus first; fall back to building from source.
find_package(pelorus QUIET)

if(pelorus_FOUND)
  # When find_package(pelorus) succeeds, use the libraries it provides
  set(PELORUS_TARGET ${pelorus_LIBRARIES})
  message(STATUS "Found installed Pelorus: ${PELORUS_TARGET}")
else()
  # Fall back to building from source
  add_subdirectory(../../pelorus_c pelorus_c_build)
  
  set(PELORUS_TARGET "")
  if(TARGET pelorus::pelorus)
    set(PELORUS_TARGET pelorus::pelorus)
  elseif(TARGET pelorus::pelorus_static)
    set(PELORUS_TARGET pelorus::pelorus_static)
  elseif(TARGET pelorus)
    set(PELORUS_TARGET pelorus)
  elseif(TARGET pelorus_static)
    set(PELORUS_TARGET pelorus_static)
  endif()

  if(PELORUS_TARGET STREQUAL "")
    message(FATAL_ERROR "pelorus library not found. Install pelorus_c (find_package(pelorus)) or keep pelorus_c as a subdirectory.")
  endif()
endif()

add_library(odometry_estimation_pelorus SHARED
  src/odometry_estimation_pelorus.cpp
)

target_link_libraries(odometry_estimation_pelorus
  glim::glim
  gtsam
  gtsam_points::gtsam_points
  ${PELORUS_TARGET}
)

install(TARGETS odometry_estimation_pelorus
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install example config (optional helper)
install(DIRECTORY config
  DESTINATION share/pelorus_glim_odometry
)

# Provide robust uninstall support.
# Note: when we build pelorus from source (add_subdirectory), that subproject may
# define an `uninstall` target first; so we always create a project-specific
# uninstall target and then hook it into the global `uninstall` if it exists.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  @ONLY
)

add_custom_target(uninstall_pelorus_glim_odometry
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
)

if(TARGET uninstall)
  add_dependencies(uninstall uninstall_pelorus_glim_odometry)
else()
  add_custom_target(uninstall DEPENDS uninstall_pelorus_glim_odometry)
endif()
