find_package(OpenMP REQUIRED)

add_executable(04_molecular_dynamics
    src/main.cpp
    src/MDSystem.cpp
    src/Scenarios.cpp
    src/WireBox.cpp
    src/MDCharts.cpp
    src/MDApp.cpp
    src/MDSim.cpp
)

target_link_libraries(04_molecular_dynamics PRIVATE
    physgl
    OpenMP::OpenMP_CXX
)
