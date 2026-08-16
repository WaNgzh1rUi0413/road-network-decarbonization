cmake_minimum_required(VERSION 3.16)
project(road_network_assignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

add_executable(prepare_network data.cpp)
add_executable(generate_od generate_od.cpp)
add_executable(user_equilibrium user_equilibrium.cpp)
target_link_libraries(prepare_network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(generate_od PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(user_equilibrium PRIVATE OpenMP::OpenMP_CXX)
