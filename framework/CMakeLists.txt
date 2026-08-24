find_package(Stb REQUIRED)

add_library(physgl STATIC
    src/Application.cpp
    src/Shader.cpp
    src/Camera.cpp
    src/Text.cpp
    src/Mesh.cpp
    src/Chart2D.cpp
    src/ParticleCloud.cpp
)

target_include_directories(physgl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(physgl PUBLIC
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(physgl PUBLIC
    glfw
    glad::glad
    glm::glm
    imgui::imgui
)

target_compile_features(physgl PUBLIC cxx_std_20)
