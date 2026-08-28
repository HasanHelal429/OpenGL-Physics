find_package(Stb REQUIRED)

add_library(physgl STATIC
    src/Application.cpp
    src/Shader.cpp
    src/ComputeShader.cpp
    src/Camera.cpp
    src/Text.cpp
    src/Mesh.cpp
    src/Chart2D.cpp
    src/ParticleCloud.cpp
    src/Deck.cpp
    src/OutputWriter.cpp
    src/GLContext.cpp
    src/HeadlessRunner.cpp
    src/Hud.cpp
    src/SimApp.cpp
)

target_include_directories(physgl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(physgl PUBLIC
    ${Stb_INCLUDE_DIR}
)

# Vendored single-header toml++ (used only by Deck.cpp; Deck.hpp stays clean).
target_include_directories(physgl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(physgl PUBLIC
    glfw
    glad::glad
    glm::glm
    imgui::imgui
)

target_compile_features(physgl PUBLIC cxx_std_20)
