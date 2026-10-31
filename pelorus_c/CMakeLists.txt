cmake_minimum_required(VERSION 3.12)
project(pelorus_c VERSION 0.0.10 LANGUAGES C)

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Options
option(PELORUS_BUILD_SHARED "Build shared library" ON)
option(PELORUS_BUILD_STATIC "Build static library" ON)
option(PELORUS_BUILD_EXAMPLES "Build C examples" OFF)
option(PELORUS_INSTALL "Install pelorus_c library and headers" ON)

# Feature flags for Rust build (match pelorus_c/Cargo.toml features)
option(PELORUS_FEATURE_RAYON "Enable Rayon CPU parallelism" ON)
option(PELORUS_FEATURE_METAL "Enable Metal acceleration" OFF)
option(PELORUS_FEATURE_CUDA "Enable CUDA acceleration" OFF)
option(PELORUS_FEATURE_VULKAN "Enable Vulkan acceleration" OFF)
option(PELORUS_FEATURE_NO_IMU_SYNC "Disable IMU synchronization" OFF)
option(PELORUS_FEATURE_LARGE_PROCESSING_QUEUE "Use larger internal processing queues" OFF)

set(PELORUS_ACCELERATION_BACKEND_COUNT 0)
foreach(BACKEND METAL CUDA VULKAN)
    if(PELORUS_FEATURE_${BACKEND})
        math(EXPR PELORUS_ACCELERATION_BACKEND_COUNT "${PELORUS_ACCELERATION_BACKEND_COUNT} + 1")
    endif()
endforeach()

if(PELORUS_ACCELERATION_BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR
        "Select at most one Pelorus acceleration backend: Metal, CUDA, or Vulkan")
endif()

if(PELORUS_FEATURE_METAL AND NOT APPLE)
    message(FATAL_ERROR "PELORUS_FEATURE_METAL is only supported on macOS")
endif()

# Find Rust/Cargo
find_program(CARGO_EXECUTABLE cargo REQUIRED)

# Capture Rust environment at configure time so it survives `sudo make install`.
# When cmake is run as the user, $HOME points to the user's home; that value is
# then baked into the generated Makefile and forwarded to every cargo invocation
# even when make is later run under sudo (which resets $HOME to /root).
if(DEFINED ENV{CARGO_HOME})
    set(PELORUS_CARGO_HOME "$ENV{CARGO_HOME}")
else()
    set(PELORUS_CARGO_HOME "$ENV{HOME}/.cargo")
endif()
if(DEFINED ENV{RUSTUP_HOME})
    set(PELORUS_RUSTUP_HOME "$ENV{RUSTUP_HOME}")
else()
    set(PELORUS_RUSTUP_HOME "$ENV{HOME}/.rustup")
endif()

# Determine build profile
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE "debug")
    set(CARGO_BUILD_FLAG "")
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/debug")
else()
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_BUILD_FLAG "--release")
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../target/release")
endif()

# Rust library output names (platform dependent)
set(RUST_SHARED_LIB "${RUST_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}pelorus${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(RUST_STATIC_LIB "${RUST_TARGET_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}pelorus${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Build feature list for Cargo
set(CARGO_FEATURES "")
if(PELORUS_FEATURE_RAYON)
    list(APPEND CARGO_FEATURES "rayon")
endif()
if(PELORUS_FEATURE_METAL)
    list(APPEND CARGO_FEATURES "metal")
endif()
if(PELORUS_FEATURE_CUDA)
    list(APPEND CARGO_FEATURES "cuda")
endif()
if(PELORUS_FEATURE_VULKAN)
    list(APPEND CARGO_FEATURES "vulkan")
endif()
if(PELORUS_FEATURE_NO_IMU_SYNC)
    list(APPEND CARGO_FEATURES "no-imu-sync")
endif()
if(PELORUS_FEATURE_LARGE_PROCESSING_QUEUE)
    list(APPEND CARGO_FEATURES "large-processing-queue")
endif()

# Join features with commas
string(REPLACE ";" "," CARGO_FEATURES_STR "${CARGO_FEATURES}")

# Build the Rust library
if(CARGO_FEATURES_STR)
    set(FEATURES_ARG "--features" "${CARGO_FEATURES_STR}")
else()
    set(FEATURES_ARG)
endif()

add_custom_target(pelorus_rust_build ALL
    COMMAND ${CMAKE_COMMAND} -E env
        "CARGO_HOME=${PELORUS_CARGO_HOME}"
        "RUSTUP_HOME=${PELORUS_RUSTUP_HOME}"
        ${CARGO_EXECUTABLE} build --package pelorus_c ${CARGO_BUILD_FLAG}
            --no-default-features ${FEATURES_ARG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    COMMENT "Building pelorus_c Rust library with Cargo"
    BYPRODUCTS 
        ${RUST_STATIC_LIB}
        ${RUST_SHARED_LIB}
)

# Define library targets
if(PELORUS_BUILD_SHARED)
    add_library(pelorus SHARED IMPORTED GLOBAL)
    # The Rust cdylib has no SONAME. Without IMPORTED_NO_SONAME the linker would
    # record the absolute build path in every consumer instead of libpelorus.so.
    set_target_properties(pelorus PROPERTIES
        IMPORTED_LOCATION ${RUST_SHARED_LIB}
        IMPORTED_NO_SONAME TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(pelorus pelorus_rust_build)
endif()

if(PELORUS_BUILD_STATIC)
    add_library(pelorus_static STATIC IMPORTED GLOBAL)
    set_target_properties(pelorus_static PROPERTIES
        IMPORTED_LOCATION ${RUST_STATIC_LIB}
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(pelorus_static pelorus_rust_build)
    
    # Link required system libraries for static linking
    if(UNIX AND NOT APPLE)
        target_link_libraries(pelorus_static INTERFACE pthread dl m)
    elseif(APPLE)
        target_link_libraries(pelorus_static INTERFACE pthread)
    elseif(WIN32)
        # Windows specific libraries
        target_link_libraries(pelorus_static INTERFACE ws2_32 userenv bcrypt ntdll)
    endif()
endif()

# Build examples if requested
if(PELORUS_BUILD_EXAMPLES)
    if(PELORUS_BUILD_SHARED)
        set(PELORUS_EXAMPLE_LIBRARY pelorus)
    elseif(PELORUS_BUILD_STATIC)
        set(PELORUS_EXAMPLE_LIBRARY pelorus_static)
    else()
        message(FATAL_ERROR
            "PELORUS_BUILD_EXAMPLES requires PELORUS_BUILD_SHARED or PELORUS_BUILD_STATIC")
    endif()
    add_subdirectory(examples)
endif()

# Installation
if(PELORUS_INSTALL)
    # Install header
    install(FILES pelorus.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Install C++ wrapper header
    install(FILES pelorus.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Install libraries
    if(PELORUS_BUILD_SHARED)
        install(FILES ${RUST_SHARED_LIB}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
    endif()
    
    if(PELORUS_BUILD_STATIC)
        install(FILES ${RUST_STATIC_LIB}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
    
    include(CMakePackageConfigHelpers)


    # Generate files into source-level build/ so packaging can include them
    set(PKG_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    file(MAKE_DIRECTORY ${PKG_OUT_DIR})

    # Generate pelorus.pc (pkg-config)
    file(WRITE "${PKG_OUT_DIR}/pelorus.pc"
      "prefix=${CMAKE_INSTALL_PREFIX}\n"
      "exec_prefix=\${prefix}\n"
      "libdir=\${prefix}/lib\n"
      "includedir=\${prefix}/include\n"
      "\n"
      "Name: Pelorus\n"
      "Description: C library bindings for Pelorus - Lidar-Inertial Odometry\n"
      "Version: ${PROJECT_VERSION}\n"
      "Libs: -L\${libdir} -lpelorus\n"
      "Libs.private: -lpthread -ldl -lm\n"
      "Cflags: -I\${includedir}\n"
    )

    # Generate pelorusConfig.cmake (self-contained) from template
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/pelorusConfig.cmake.in"
      "${PKG_OUT_DIR}/pelorusConfig.cmake"
      @ONLY
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/pelorusConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    install(FILES
        "${PKG_OUT_DIR}/pelorusConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pelorusConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pelorus
    )

    # Install pkg-config file (prefer generated build/ file)
    if(EXISTS "${PKG_OUT_DIR}/pelorus.pc")
      install(FILES "${PKG_OUT_DIR}/pelorus.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
      )
    else()
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pelorus.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
      )
    endif()

    # Uninstall target (uses install_manifest.txt generated by the install step).
    # A parent project such as the ament package brings its own.
    if(NOT TARGET uninstall)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            @ONLY
        )
        add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            COMMENT "Uninstalling files listed in install_manifest.txt"
        )
    endif()
endif()
