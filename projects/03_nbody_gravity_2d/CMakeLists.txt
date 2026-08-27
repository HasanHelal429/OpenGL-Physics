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
