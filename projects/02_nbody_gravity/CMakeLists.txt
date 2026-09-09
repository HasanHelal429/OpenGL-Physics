find_package(OpenMP REQUIRED)

add_executable(02_nbody_gravity
    src/main.cpp
    src/NBodySystem.cpp
    src/Octree.cpp
    src/AdaptiveFmm.cpp
    src/SphericalHarmonics.cpp
    src/SphericalFmm.cpp
    src/Scenarios.cpp
    src/DriftChart.cpp
    src/DiagnosticsWorker.cpp
    src/ScalingSweepWorker.cpp
    src/NBodyApp.cpp
)

target_link_libraries(02_nbody_gravity PRIVATE
    physgl
    OpenMP::OpenMP_CXX
)

# Headless scaling benchmark. Links the solver translation units plus glm and
# OpenMP only -- no physgl, no GLFW, no GL context -- so it runs unchanged
# inside a batch job, which is where these timings are worth taking.
add_executable(02_nbody_scaling
    src/bench_main.cpp
    src/NBodySystem.cpp
    src/Octree.cpp
    src/AdaptiveFmm.cpp
    src/SphericalHarmonics.cpp
    src/SphericalFmm.cpp
    src/Scenarios.cpp
)

target_include_directories(02_nbody_scaling PRIVATE src)

target_link_libraries(02_nbody_scaling PRIVATE
    glm::glm
    OpenMP::OpenMP_CXX
)

target_compile_features(02_nbody_scaling PRIVATE cxx_std_20)
