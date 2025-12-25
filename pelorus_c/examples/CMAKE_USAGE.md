# Example CMakeLists.txt for using Pelorus C library

This directory demonstrates how to use the Pelorus C library in your own CMake project.

## Method 1: Using find_package (after installation)

```cmake
cmake_minimum_required(VERSION 3.12)
project(my_pelorus_app)

# Find the Pelorus library
find_package(pelorus REQUIRED)

add_executable(my_app main.c)
target_link_libraries(my_app pelorus::pelorus)
```

## Method 2: Using pkg-config

```cmake
cmake_minimum_required(VERSION 3.12)
project(my_pelorus_app)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PELORUS REQUIRED pelorus)

add_executable(my_app main.c)
target_include_directories(my_app PRIVATE ${PELORUS_INCLUDE_DIRS})
target_link_libraries(my_app ${PELORUS_LIBRARIES})
```

## Method 3: Direct subdirectory inclusion (development)

If you haven't installed the library, you can include it as a subdirectory:

```cmake
cmake_minimum_required(VERSION 3.12)
project(my_pelorus_app)

# Add pelorus_c as a subdirectory
add_subdirectory(path/to/pelorus/pelorus_c)

add_executable(my_app main.c)
target_link_libraries(my_app pelorus)
```

## Building the examples

From this directory:

```bash
mkdir build
cd build
cmake ..
make
./my_pelorus_app
```

## C++ Usage

For C++ projects, include `pelorus.hpp` instead:

```cpp
#include <pelorus.hpp>

int main() {
    pelorus::Pelorus pelorus;
    // Use the C++ wrapper...
    return 0;
}
```
