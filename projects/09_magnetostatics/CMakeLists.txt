add_executable(09_magnetostatics
    src/main.cpp
    src/FieldSolver.cpp
    src/Sources.cpp
    src/MagnetostaticsSim.cpp
)

target_link_libraries(09_magnetostatics PRIVATE physgl)
