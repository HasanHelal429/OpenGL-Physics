find_package(OpenMP REQUIRED)

add_executable(03_nbody_gravity_2d
    src/main.cpp
    src/NBodySystem.cpp
    src/Quadtree.cpp
    src/ComplexFmm.cpp
    src/ComplexFmmTree.cpp
    src/Scenarios.cpp
    src/DriftChart.cpp
    src/NBodyApp.cpp
)

target_link_libraries(03_nbody_gravity_2d PRIVATE
    physgl
    OpenMP::OpenMP_CXX
)

# Headless scaling benchmark -- see 02_nbody_gravity/CMakeLists.txt for why
# this target deliberately avoids physgl.
add_executable(03_nbody_scaling_2d
    src/bench_main.cpp
    src/NBodySystem.cpp
    src/Quadtree.cpp
    src/ComplexFmm.cpp
    src/ComplexFmmTree.cpp
    src/Scenarios.cpp
)

target_include_directories(03_nbody_scaling_2d PRIVATE src)

target_link_libraries(03_nbody_scaling_2d PRIVATE
    glm::glm
    OpenMP::OpenMP_CXX
)

target_compile_features(03_nbody_scaling_2d PRIVATE cxx_std_20)
