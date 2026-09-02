add_executable(08_compressible_fluid
    src/main.cpp
    src/Euler1D.cpp
    src/CompressibleSim.cpp
)

target_link_libraries(08_compressible_fluid PRIVATE physgl)
