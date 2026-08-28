add_executable(05_tdse_gpu
    src/main.cpp
    src/TdseField.cpp
    src/TdseSim.cpp
)

target_link_libraries(05_tdse_gpu PRIVATE physgl)
