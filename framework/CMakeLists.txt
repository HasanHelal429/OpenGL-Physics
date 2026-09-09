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

# Headless OpenGL without a display server (EGL_PLATFORM_DEVICE_EXT).
# GLContext's hidden-GLFW-window path needs X11/Wayland, which a batch node
# does not have; with EGL, CreateHidden() gets a real GPU context anyway.
# See framework/include/framework/GLContext.hpp.
if(UNIX AND NOT APPLE)
    find_package(OpenGL QUIET COMPONENTS EGL)
    if(TARGET OpenGL::EGL)
        target_link_libraries(physgl PUBLIC OpenGL::EGL)
        target_compile_definitions(physgl PUBLIC PHYSGL_HAVE_EGL)
        message(STATUS "physgl: EGL headless backend ENABLED")
    else()
        message(STATUS "physgl: EGL not found -- headless runs will require a display server")
    endif()
endif()
