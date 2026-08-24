find_package(Eigen3 CONFIG REQUIRED)
find_package(spectra CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)

add_executable(01_hartree_fock
    src/main.cpp
    src/Shells.cpp
    src/Potentials.cpp
    src/RadialEigensolver.cpp
    src/ScfSolver.cpp
    src/ScfWorker.cpp
    src/HFVisualizerApp.cpp
)

target_link_libraries(01_hartree_fock PRIVATE
    physgl
    Eigen3::Eigen
    Spectra::Spectra
    implot::implot
)
