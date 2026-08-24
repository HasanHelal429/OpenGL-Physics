add_library(physgl STATIC
    src/Application.cpp
    src/Shader.cpp
    src/Camera.cpp
)

target_include_directories(physgl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(physgl PUBLIC
    glfw
    glad::glad
    glm::glm
    imgui::imgui
)

target_compile_features(physgl PUBLIC cxx_std_20)
