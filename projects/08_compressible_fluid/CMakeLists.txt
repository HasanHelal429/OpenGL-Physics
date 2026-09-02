add_executable(08_compressible_fluid
    src/main.cpp
    src/Euler1D.cpp
    src/Euler2D.cpp
    src/CompressibleSim.cpp
    src/CompressibleSim2D.cpp
)

target_link_libraries(08_compressible_fluid PRIVATE physgl)
