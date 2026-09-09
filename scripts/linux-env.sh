#!/usr/bin/env bash
# Environment for the Linux build. Source it, don't run it:
#
#   source scripts/linux-env.sh
#   cmake --preset linux-release && cmake --build --preset linux-release
#
# Written for NERSC Perlmutter (SUSE + PrgEnv-gnu + an NVIDIA GPU), but it only
# assumes: a C++20 g++, cmake >= 3.21, ninja, a vcpkg checkout, and EGL. Both
# knobs below can be overridden before sourcing.
#
# Why the python shim: vcpkg builds part of the X11 stack with meson, and meson
# needs python >= 3.7. Perlmutter's /usr/bin/python3 is 3.6.15, so we put a
# newer interpreter first on PATH just for the build.

# --- vcpkg ------------------------------------------------------------------
: "${VCPKG_ROOT:=${SCRATCH:-$HOME}/vcpkg}"
export VCPKG_ROOT

if [ ! -x "$VCPKG_ROOT/vcpkg" ]; then
    echo "linux-env: no vcpkg at $VCPKG_ROOT" >&2
    echo "linux-env: git clone https://github.com/microsoft/vcpkg \"$VCPKG_ROOT\"" >&2
    echo "linux-env: \"$VCPKG_ROOT\"/bootstrap-vcpkg.sh -disableMetrics" >&2
else
    export PATH="$VCPKG_ROOT:$PATH"
fi

# --- python >= 3.7 for meson-based vcpkg ports -------------------------------
: "${PHYSGL_TOOLS:=${SCRATCH:-$HOME}/.physgl-tools}"

physgl_py_ok() {
    command -v python3 >/dev/null 2>&1 &&
        python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)' 2>/dev/null
}

if ! physgl_py_ok; then
    physgl_newer=""
    for cand in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8; do
        if command -v "$cand" >/dev/null 2>&1; then
            physgl_newer=$(command -v "$cand")
            break
        fi
    done
    if [ -n "$physgl_newer" ]; then
        mkdir -p "$PHYSGL_TOOLS/bin"
        ln -sf "$physgl_newer" "$PHYSGL_TOOLS/bin/python3"
        ln -sf "$physgl_newer" "$PHYSGL_TOOLS/bin/python"
        export PATH="$PHYSGL_TOOLS/bin:$PATH"
        echo "linux-env: python3 -> $physgl_newer (for meson)"
    else
        echo "linux-env: WARNING no python >= 3.7 found; vcpkg meson ports will fail" >&2
    fi
fi
unset -f physgl_py_ok

# --- headless GL -------------------------------------------------------------
# With no DISPLAY, GLContext::CreateHidden() takes its EGL path automatically.
# Force either backend with PHYSGL_GL_BACKEND=egl|glfw; pick a specific GPU
# with PHYSGL_EGL_DEVICE=<index>.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "linux-env: no display server -- headless runs will use EGL"
fi

echo "linux-env: VCPKG_ROOT=$VCPKG_ROOT"
echo "linux-env: $(command -v g++) $(g++ -dumpversion 2>/dev/null)"
echo "linux-env: $(cmake --version 2>/dev/null | head -1)"
