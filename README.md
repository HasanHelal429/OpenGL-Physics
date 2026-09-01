# OpenGL Physics

C++/OpenGL numerical physics simulators — a faster, visually interactive
counterpart to the [`Physics Simulations`](../Physics%20Simulations) Python
project.

## Toolchain

- Compiler: MinGW-w64 GCC (via MSYS2, `C:\msys64\mingw64\bin`)
- Build: CMake + Ninja
- Packages: vcpkg (manifest mode, `vcpkg.json`), triplet `x64-mingw-dynamic`
- Libraries: GLFW3 (windowing/input), glad2 (OpenGL 4.6 core loader), GLM (math), Dear ImGui (debug UI, `00`–`04` only), stb; toml++ vendored under `framework/third_party/`

`VCPKG_ROOT` is set to `C:\vcpkg` as a user environment variable; CMake picks
it up automatically via the presets below.

## Layout

```
framework/          Shared library ("physgl"):
                     Application  window + fixed-timestep loop + input
                     Shader / ComputeShader / Camera / Text / Chart2D / ...
                     Simulation   the interface batch/interactive sims implement
                     Deck         TOML input deck (vendored toml++)
                     OutputWriter .npy frames + diagnostics.csv + manifest.json
                     GLContext / HeadlessRunner   no-window batch runner
                     SimApp / Hud a window + a free-floating control cluster (no ImGui)
projects/            One executable per simulation. Numbered by creation order.
  00_hello_triangle/  Sanity-check — spinning triangle + ImGui overlay.
  01_hartree_fock/    Atomic Hartree-Fock-Slater / Kohn-Sham LDA SCF solver.
  02/03_nbody_gravity Direct + Barnes-Hut N-body (3D / 2D-with-FMM).
  04_molecular_dynamics  3D periodic Lennard-Jones MD -- Nose-Hoover/Berendsen
                       thermostats, Berendsen NPT barostat, deck-driven
                       melting/freezing temperature ramp, optional
                       Kob-Andersen binary glass mixture. Both models: the
                       original ImGui + live-chart app, and (preferred) the
                       Simulation/Deck/headless model -- see its README.
  05_tdse_gpu/        2D time-dependent Schrödinger, split-step Fourier on the
                       GPU (hand-written FFT compute shader). First project on
                       the Simulation/Deck/headless model — see its README.
  06_tidal_disruption/ Tidal disruption event, built up tier by tier. Tier 1
                       (current): self-gravitating SPH star (GPU compute
                       shaders) in hydrostatic equilibrium, no black hole yet
                       — see its README.
```

Each simulation project is its own executable linked against `physgl`, so
projects stay independent while sharing the windowing/render/shader boilerplate.

### Two ways a project can run

Older projects (`01`–`03`) subclass `fw::Application` directly, with an ImGui
control panel and live `Chart2D` plots. `04_molecular_dynamics` keeps that
original app (`MDApp`) alongside a second, deck-driven `fw::Simulation`
implementation (`MDSim`) added later -- see its README for why both exist.

From `05` on, the preferred model is `fw::Simulation` + an **input deck**:

- **batch** — `sim --deck run.toml --out results/` runs headless and writes raw
  data (`.npy` field frames, `diagnostics.csv`, `manifest.json`) that the
  project's `tools/*.py` turn into movies and validation plots. This is the
  default for stiff / large-Δt / non-time-evolution solvers.
- **interactive** — `fw::SimApp` opens a window with mouse camera control and a
  small free-floating HUD (play/pause, step, reset, speed, record, `F12`
  screenshot) drawn as real OpenGL geometry, no docked panel.

The `Application` loop separates the physics step from the render step:
`OnFixedUpdate(dt)` runs at a fixed rate (default 120 Hz) for numerical
integration, while `OnUpdate`/`OnRender` run once per frame.

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

1. `projects/NN_name/src/` — either subclass `fw::Application` directly (like
   `00`–`04`), or implement `fw::Simulation` and let `main.cpp` dispatch to
   `fw::RunHeadless` / `fw::SimApp` (like `05`, preferred).
2. `projects/NN_name/CMakeLists.txt`:
   ```cmake
   add_executable(NN_name src/main.cpp ...)
   target_link_libraries(NN_name PRIVATE physgl)
   ```
3. Add `add_subdirectory(NN_name)` to `projects/CMakeLists.txt`.
