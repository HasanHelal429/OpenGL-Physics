add_executable(10_fdtd
    src/main.cpp
    src/Fdtd2D.cpp
)

target_link_libraries(10_fdtd PRIVATE physgl)
