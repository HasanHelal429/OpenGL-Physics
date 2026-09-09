find_package(OpenMP REQUIRED)

# Shared numerics for 02_nbody_gravity and 03_nbody_gravity_2d: SoA state, the
# flat adaptive tree, the force solvers, the leapfrog integrator, scenarios
# and diagnostics. Dimension-generic (template<int D>), explicitly
# instantiated for D=2 and D=3.
add_library(nbody_core STATIC
    src/Tree.cpp
    src/Solvers.cpp
    src/MutualFmm.cpp
    src/Integrator.cpp
    src/Scenarios.cpp
    src/DeckSim.cpp
    src/SelfTest.cpp
    src/gpu/GpuDirect.cpp
    src/gpu/GpuBarnesHut.cpp
    src/gpu/GpuQuadtree.cpp
)

target_include_directories(nbody_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nbody_core PUBLIC
    physgl                     # fw::Simulation / Deck / OutputWriter / ParticleCloud
    glm::glm
    OpenMP::OpenMP_CXX
)

target_compile_features(nbody_core PUBLIC cxx_std_20)

# Numerics kernels benefit from aggressive optimisation. -ffast-math is
# deliberately NOT enabled here: the integrator and the energy/angular-momentum
# reductions must stay strict IEEE for the Kepler/Lagrange conservation gates.
# A -ffast-math kernels object library is a later phase (see the plan doc).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nbody_core PRIVATE -O3 -march=native -funroll-loops)
endif()

# Standalone numerics validation (no GL). Same checks as 02/03's --selftest.
add_executable(nbody_core_selftest test/main.cpp)
target_link_libraries(nbody_core_selftest PRIVATE nbody_core OpenMP::OpenMP_CXX)
