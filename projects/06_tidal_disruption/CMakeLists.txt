add_executable(06_tidal_disruption
    src/main.cpp
    src/TdeSim.cpp
)

target_link_libraries(06_tidal_disruption PRIVATE physgl)
