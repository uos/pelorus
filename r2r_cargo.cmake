# #
# For r2r 0.9.7
#
# cmake code for simple colcon integration.
# See https://github.com/m-dahl/r2r_minimal_node/
# for an example of how to use it to build with colcon.
#

# Need at least 3.21 for IMPORTED_TARGETS property that we rely on.
# NOTE! Ubuntu 20.04 comes 3.16.3 by default, so to use this you need
# to update cmake. This can be done by adding the kitware apt repo:
# https://apt.kitware.com
cmake_minimum_required(VERSION "3.21")

# traverse dependencies for all message packages.
function(get_idl_deps OUT_PKG_DIRS PKG)
    find_package(${PKG} REQUIRED)
    target_link_libraries(dummy ${${PKG}_LIBRARIES})
    include_directories(dummy ${${PKG}_INCLUDE_DIRS})

    list(APPEND VISITED_TARGETS ${PKG})

    # only keep track of packages that include idl files
    set(PKG_DIRS "")
    if(DEFINED ${PKG}_IDL_FILES)
        list(APPEND PKG_DIRS ${${PKG}_DIR})
    endif()

    foreach(LIB ${${PKG}_DEPENDENCIES})
        list(FIND VISITED_TARGETS ${LIB} VISITED)
        if (${VISITED} EQUAL -1)
            get_idl_deps(NEW_PKG_DIRS ${LIB})
            list(APPEND PKG_DIRS ${NEW_PKG_DIRS})
            list(REMOVE_DUPLICATES PKG_DIRS)
        endif()
    endforeach()
    set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
    set(${OUT_PKG_DIRS} ${PKG_DIRS} PARENT_SCOPE)
endfunction()

function(r2r_cargo)
  # pretend that we want to compile c code to get all library paths etc...
  add_executable (dummy EXCLUDE_FROM_ALL dummy.c)

  # traverse list of wanted packages to add dependencies
  foreach(f ${ARGN})
    find_package(${f} REQUIRED)
    set(REC_PKG_DIRS "")
    get_idl_deps(REC_PKG_DIRS ${f})
    list(APPEND CMAKE_IDL_PACKAGES "${REC_PKG_DIRS}")
  endforeach()

  list(REMOVE_DUPLICATES CMAKE_IDL_PACKAGES)
  string (REPLACE ";" ":" CMAKE_IDL_PACKAGES_STR "${CMAKE_IDL_PACKAGES}")
  set(ENV{CMAKE_IDL_PACKAGES} ${CMAKE_IDL_PACKAGES_STR})

  # On OSX colcon eats the DYLD_LIBRARY_PATH... so we need to add the rpaths
  # manually...
  set(RUSTFLAGS "")

  # get imported libs
  get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)

  foreach(p ${importTargets})
    get_property(_LIBLOC TARGET "${p}" PROPERTY LOCATION)
    if(DEFINED _LIBLOC)
      list(APPEND CMAKE_LIBRARIES "${_LIBLOC}")
      get_filename_component(_PARENT "${_LIBLOC}" DIRECTORY)
      if(IS_DIRECTORY ${_PARENT})
          list(APPEND RUSTFLAGS "-C link-arg=-Wl,-rpath,${_PARENT}")
      endif()
    endif()
  endforeach()

  list(REMOVE_DUPLICATES RUSTFLAGS)
  string (REPLACE ";" " " RUSTFLAGS_STR "${RUSTFLAGS}")
  set(ENV{RUSTFLAGS} ${RUSTFLAGS_STR})

  # get include paths
  get_property(includeDirs DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES includeDirs)
  string (REPLACE ";" ":" CMAKE_INCLUDE_DIRS_STR "${includeDirs}")
  set(ENV{CMAKE_INCLUDE_DIRS} ${CMAKE_INCLUDE_DIRS_STR})

  set(FEATURES "ros2c")

  # Master switch to disable all defaults at once (defaults to ON)
  if(NOT DEFINED PELORUS_ALL_DEFAULTS OR PELORUS_ALL_DEFAULTS)
    if(NOT DEFINED PELORUS_RAT OR PELORUS_RAT)
      list(APPEND FEATURES "rat")
      list(APPEND FEATURES "rayon")
    endif()

  endif()

  # handle cmake args and env args
  if(PELORUS_CUDA)
    list(APPEND FEATURES "cuda")
  elseif(DEFINED ENV{PELORUS_CUDA}
         AND NOT "$ENV{PELORUS_CUDA}" STREQUAL ""
         AND NOT "$ENV{PELORUS_CUDA}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_CUDA}" STREQUAL "OFF")
    list(APPEND FEATURES "cuda")
  endif()
  if(PELORUS_METAL)
    list(APPEND FEATURES "metal")
  elseif(DEFINED ENV{PELORUS_METAL}
         AND NOT "$ENV{PELORUS_METAL}" STREQUAL ""
         AND NOT "$ENV{PELORUS_METAL}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_METAL}" STREQUAL "OFF")
    list(APPEND FEATURES "metal")
  endif()
  if(PELORUS_VULKAN)
    list(APPEND FEATURES "vulkan")
  elseif(DEFINED ENV{PELORUS_VULKAN}
         AND NOT "$ENV{PELORUS_VULKAN}" STREQUAL ""
         AND NOT "$ENV{PELORUS_VULKAN}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_VULKAN}" STREQUAL "OFF")
    list(APPEND FEATURES "vulkan")
  endif()
  if(PELORUS_MIMALLOC)
    list(APPEND FEATURES "mimalloc")
  endif()
  if(PELORUS_RAYON)
    list(APPEND FEATURES "rayon")
  endif()
  if(PELORUS_MAP)
    list(APPEND FEATURES "map")
  endif()
  if (PELORUS_NO_VIZ)
    list(APPEND FEATURES "no-viz")
  endif()
  if (PELORUS_RERUN)
    list(APPEND FEATURES "rerun")
  endif()
  if(PELORUS_LARGE_PROCESSING_QUEUE)
    list(APPEND FEATURES "large-processing-queue")
  elseif(DEFINED ENV{PELORUS_LARGE_PROCESSING_QUEUE}
         AND NOT "$ENV{PELORUS_LARGE_PROCESSING_QUEUE}" STREQUAL ""
         AND NOT "$ENV{PELORUS_LARGE_PROCESSING_QUEUE}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_LARGE_PROCESSING_QUEUE}" STREQUAL "OFF")
    list(APPEND FEATURES "large-processing-queue")
  endif()
  if(PELORUS_LRU_EVICTION)
    list(APPEND FEATURES "lru-eviction")
  elseif(DEFINED ENV{PELORUS_LRU_EVICTION}
         AND NOT "$ENV{PELORUS_LRU_EVICTION}" STREQUAL ""
         AND NOT "$ENV{PELORUS_LRU_EVICTION}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_LRU_EVICTION}" STREQUAL "OFF")
    list(APPEND FEATURES "lru-eviction")
  endif()
  if(PELORUS_PLAIN_UPDATE)
    list(APPEND FEATURES "plain-update")
  elseif(DEFINED ENV{PELORUS_PLAIN_UPDATE}
         AND NOT "$ENV{PELORUS_PLAIN_UPDATE}" STREQUAL ""
         AND NOT "$ENV{PELORUS_PLAIN_UPDATE}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_PLAIN_UPDATE}" STREQUAL "OFF")
    list(APPEND FEATURES "plain-update")
  endif()
  if(PELORUS_NO_MATURITY_DECAY)
    list(APPEND FEATURES "no-maturity-decay")
  elseif(DEFINED ENV{PELORUS_NO_MATURITY_DECAY}
         AND NOT "$ENV{PELORUS_NO_MATURITY_DECAY}" STREQUAL ""
         AND NOT "$ENV{PELORUS_NO_MATURITY_DECAY}" STREQUAL "0"
         AND NOT "$ENV{PELORUS_NO_MATURITY_DECAY}" STREQUAL "OFF")
    list(APPEND FEATURES "no-maturity-decay")
  endif()


  string(JOIN "," FEATURES_STR ${FEATURES})

  # optim, only with local builds
  # set(ENV{RUSTFLAGS} "-C target-cpu=native -C llvm-args=-cost-kind=latency")
 
  add_custom_target(cargo_target ALL
      COMMAND ${CMAKE_COMMAND} "-E" "env" "--unset=MAKEFLAGS" "--unset=MFLAGS" "RUSTFLAGS=$ENV{RUSTFLAGS}" "CMAKE_IDL_PACKAGES=$ENV{CMAKE_IDL_PACKAGES}" "CMAKE_INCLUDE_DIRS=$ENV{CMAKE_INCLUDE_DIRS}" "cargo" "build" "-p" "pelorus_cli" "--profile" "colcon" "--no-default-features" "--features" "${FEATURES_STR}" "--locked"
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      )
  unset(CARGO_CLEAN CACHE)

endfunction()
