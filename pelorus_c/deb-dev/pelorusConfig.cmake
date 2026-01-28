# Compute package prefix dir from this file's install location. The config
# is installed to: <prefix>/<libdir>/cmake/pelorus, so go up two levels.
get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

include(CMakeFindDependencyMacro)

# Prefer common include dir under the prefix
set(pelorus_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

# Library file layout: assume libraries live in the parent dir of this
# cmake config directory (i.e., <prefix>/<libdir>/)
set(_pelorus_shared "${CMAKE_CURRENT_LIST_DIR}/../${CMAKE_SHARED_LIBRARY_PREFIX}pelorus${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_pelorus_static "${CMAKE_CURRENT_LIST_DIR}/../${CMAKE_STATIC_LIBRARY_PREFIX}pelorus${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Try to find pelorus library if not at expected install locations
find_library(_pelorus_lib NAMES pelorus HINTS "${CMAKE_CURRENT_LIST_DIR}/.." NO_DEFAULT_PATH)
if(NOT _pelorus_lib)
  find_library(_pelorus_lib NAMES pelorus)
endif()

# Shared target
if((EXISTS "${_pelorus_shared}" OR _pelorus_lib) AND NOT TARGET pelorus::pelorus)
  add_library(pelorus::pelorus SHARED IMPORTED)
  if(EXISTS "${_pelorus_shared}")
    set_target_properties(pelorus::pelorus PROPERTIES IMPORTED_LOCATION "${_pelorus_shared}")
  elseif(_pelorus_lib)
    set_target_properties(pelorus::pelorus PROPERTIES IMPORTED_LOCATION "${_pelorus_lib}")
  endif()
  set_target_properties(pelorus::pelorus PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${pelorus_INCLUDE_DIRS}")
endif()

# Static target
if((EXISTS "${_pelorus_static}") AND NOT TARGET pelorus::pelorus_static)
  add_library(pelorus::pelorus_static STATIC IMPORTED)
  set_target_properties(pelorus::pelorus_static PROPERTIES
    IMPORTED_LOCATION "${_pelorus_static}"
    INTERFACE_INCLUDE_DIRECTORIES "${pelorus_INCLUDE_DIRS}"
  )

  # Match pelorus_c/CMakeLists.txt: link required system libs when using the static archive.
  if(UNIX AND NOT APPLE)
    find_dependency(Threads)
    target_link_libraries(pelorus::pelorus_static INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
  elseif(APPLE)
    find_dependency(Threads)
    target_link_libraries(pelorus::pelorus_static INTERFACE Threads::Threads)
  elseif(WIN32)
    target_link_libraries(pelorus::pelorus_static INTERFACE ws2_32 userenv bcrypt ntdll)
  endif()
endif()

# Convenience variable: prefer shared if present, otherwise static.
if(TARGET pelorus::pelorus)
  set(pelorus_LIBRARIES pelorus::pelorus)
elseif(TARGET pelorus::pelorus_static)
  set(pelorus_LIBRARIES pelorus::pelorus_static)
elseif(_pelorus_lib)
  set(pelorus_LIBRARIES "${_pelorus_lib}")
endif()
