# OpenGL Physics

C++/OpenGL numerical physics simulators — a faster, visually interactive
counterpart to the [`Physics Simulations`](../Physics%20Simulations) Python
project.

## Toolchain

- Compiler: MinGW-w64 GCC (via MSYS2, `C:\msys64\mingw64\bin`)
- Build: CMake + Ninja
- Packages: vcpkg (manifest mode, `vcpkg.json`), triplet `x64-mingw-dynamic`
- Libraries: GLFW3 (windowing/input), glad2 (OpenGL 4.6 core loader), GLM (math), Dear ImGui (debug UI)

`VCPKG_ROOT` is set to `C:\vcpkg` as a user environment variable; CMake picks
it up automatically via the presets below.

## Layout

```
framework/          Shared library ("physgl"): Application (window + fixed-
                     timestep loop + input), Shader, Camera.
projects/            One executable per simulation. Numbered by creation order.
  00_hello_triangle/  Sanity-check project — spinning triangle + ImGui overlay.
```

Each simulation project is its own executable linked against `physgl`, so
projects stay independent while sharing the windowing/render/shader boilerplate.

The `Application` loop separates the physics step from the render step:
`OnFixedUpdate(dt)` runs at a fixed rate (default 120 Hz) for numerical
integration, while `OnUpdate`/`OnRender` run once per frame. This keeps
simulations numerically stable regardless of display frame rate.

## Build & run

```sh
cmake --preset debug
cmake --build --preset debug
./build/debug/projects/00_hello_triangle/00_hello_triangle.exe
```

Use `--preset release` for an optimized build. vcpkg builds dependencies
from source on first configure, which can take a while.

## Adding a new project

1. `projects/NN_name/src/main.cpp` — subclass `fw::Application`, override
   `OnStart`/`OnFixedUpdate`/`OnRender`/`OnImGui` as needed.
2. `projects/NN_name/CMakeLists.txt`:
   ```cmake
   add_executable(NN_name src/main.cpp)
   target_link_libraries(NN_name PRIVATE physgl)
   ```
3. Add `add_subdirectory(NN_name)` to `projects/CMakeLists.txt`.
