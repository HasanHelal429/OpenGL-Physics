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
  01_hartree_fock/    Atomic Hartree-Fock-Slater / Kohn-Sham LDA SCF solver, ported
                       from Physics Simulations/Quantum Mechanics/HF_solver/, with
                       a live ImPlot view of density/potential/orbital energies/
                       convergence as the SCF loop runs on a background thread.
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

**For `01_hartree_fock` specifically, use the `release` preset.** Its SCF
loop leans on Eigen/Spectra, whose header-only template code is dramatically
slower without optimizations — e.g. Argon's SCF run went from single-digit
seconds in Release to multiple minutes in Debug during development. Debug is
fine for setting breakpoints, not for actually running a solve.

Both presets need `C:\msys64\mingw64\bin` on `PATH` (added to the user
env var during setup) and `VCPKG_ROOT=C:\vcpkg` — a terminal opened before
that env change won't see them until restarted.

## Adding a new project

1. `projects/NN_name/src/main.cpp` — subclass `fw::Application`, override
   `OnStart`/`OnFixedUpdate`/`OnRender`/`OnImGui` as needed.
2. `projects/NN_name/CMakeLists.txt`:
   ```cmake
   add_executable(NN_name src/main.cpp)
   target_link_libraries(NN_name PRIVATE physgl)
   ```
3. Add `add_subdirectory(NN_name)` to `projects/CMakeLists.txt`.
