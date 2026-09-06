add_executable(12_tddft
    src/main.cpp
    src/Tddft3D.cpp
    src/Tddft3DSim.cpp
)

target_link_libraries(12_tddft PRIVATE physgl)
