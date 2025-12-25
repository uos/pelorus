cmake_minimum_required(VERSION 3.16)
project(pelorus_glim_frontend LANGUAGES CXX)

find_package(glim REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "Default install prefix" FORCE)
endif()

# Try to find installed pelorus first; fall back to building from source.
find_package(pelorus QUIET)
if(NOT pelorus_FOUND)
  add_subdirectory(../../pelorus_c pelorus_c_build)
endif()

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

add_library(pelorus_glim_frontend SHARED
  src/pelorus_glim_frontend.cpp
)

target_link_libraries(pelorus_glim_frontend
  glim::glim
  ${PELORUS_TARGET}
)

install(TARGETS pelorus_glim_frontend
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
