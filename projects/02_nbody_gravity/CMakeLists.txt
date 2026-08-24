find_package(OpenMP REQUIRED)

add_executable(02_nbody_gravity
    src/main.cpp
    src/NBodySystem.cpp
    src/Octree.cpp
    src/Scenarios.cpp
    src/DriftChart.cpp
    src/NBodyApp.cpp
)

target_link_libraries(02_nbody_gravity PRIVATE
    physgl
    OpenMP::OpenMP_CXX
)
