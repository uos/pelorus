cmake_minimum_required(VERSION 3.5)
project(pelorus)

find_package(ament_cmake REQUIRED)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

include(r2r_cargo.cmake)

r2r_cargo(rcl rcl_action sensor_msgs nav_msgs tf2_msgs visualization_msgs)

if(WIN32)
  set(SUFFIX ".exe")
else()
  set(SUFFIX "")
endif()

install(PROGRAMS
  ${CMAKE_SOURCE_DIR}/target/colcon/${PROJECT_NAME}${SUFFIX}
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  ${CMAKE_SOURCE_DIR}/target/colcon/${PROJECT_NAME}${SUFFIX}
  DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY pelorus_cli/rviz
  DESTINATION share/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Optional GLIM odometry plugin.
#
# GLIM, GTSAM and gtsam_points are heavy system dependencies that most people
# building the ROS node do not have installed. So they are probed for instead of
# required: when they are all present the plugin is built and installed into the
# colcon prefix alongside the node, and when they are not the node still builds
# exactly as before and we only print a note saying what was missing.
# ---------------------------------------------------------------------------
option(PELORUS_BUILD_GLIM
  "Build the GLIM odometry plugin when GLIM and its dependencies are found" ON)

if(PELORUS_BUILD_GLIM)
  find_package(glim QUIET)
  find_package(GTSAM 4.2 QUIET)
  find_package(gtsam_points 1.1.0 QUIET)

  set(PELORUS_GLIM_MISSING "")
  if(NOT glim_FOUND)
    list(APPEND PELORUS_GLIM_MISSING "glim")
  endif()
  if(NOT GTSAM_FOUND)
    list(APPEND PELORUS_GLIM_MISSING "GTSAM >= 4.2")
  endif()
  if(NOT gtsam_points_FOUND)
    list(APPEND PELORUS_GLIM_MISSING "gtsam_points >= 1.1.0")
  endif()

  if(PELORUS_GLIM_MISSING STREQUAL "")
    message(STATUS
      "GLIM dependencies found, building the pelorus GLIM odometry plugin")
    add_subdirectory(glim_integration/pelorus_glim_odometry)
  else()
    string(REPLACE ";" ", " PELORUS_GLIM_MISSING_STR "${PELORUS_GLIM_MISSING}")
    message(STATUS
      "GLIM odometry plugin skipped, not found: ${PELORUS_GLIM_MISSING_STR}. "
      "The pelorus ROS node builds as usual. Install these to get the plugin, "
      "or pass -DPELORUS_BUILD_GLIM=OFF to skip this check.")
  endif()
endif()

ament_package()
