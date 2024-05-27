# Note: this CMakeLists.txt is only for development without Rust. The build.rs uses cc directly without cmake.
cmake_minimum_required(VERSION 3.5)
project(lio_icp)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

add_library(lio_icp src/icp.cpp)

target_include_directories(lio_icp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    small_gicp/include
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(lio_icp
    ${TBB_LIBRARIES}
)

# Install
install(TARGETS lio_icp
    EXPORT lio_icp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)