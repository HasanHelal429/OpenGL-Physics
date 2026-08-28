add_executable(07_grhd
    src/main.cpp
    src/GrhdSim.cpp
    src/SchwarzschildSim.cpp
    src/KerrEquatorialSim.cpp
)

target_link_libraries(07_grhd PRIVATE physgl)
