# Overlay of upstream vcpkg's glfw3 port.
#
# Only difference: the `libx11` dependency is dropped from vcpkg.json so GLFW
# links the X11 development libraries already installed on the system rather
# than making vcpkg rebuild the whole X.org stack from source. On an HPC login
# node that rebuild is both unnecessary (the -devel packages are there) and
# broken -- some of those ports need `autoconf-archive` and a newer meson than
# the image provides.
#
# Keep in sync with ${VCPKG_ROOT}/ports/glfw3/portfile.cmake when bumping.

if (VCPKG_TARGET_IS_EMSCRIPTEN)
    # emscripten has built-in glfw3 library
    set(VCPKG_BUILD_TYPE release)
    file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/glfw3Config.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/glfw3")
    set(VCPKG_POLICY_EMPTY_PACKAGE enabled)
    return()
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO glfw/glfw
    REF "3.5.1"
    SHA512 42e8b8bdc2ecaab8d33d43cab6cf6e4bfd1f23492a5f13ec4b3762b74661dc3c9625721f436fbf841909b9af3ac6dcb8080201b36592d33c2bb196f768fc307f
    HEAD_REF master
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
    wayland         GLFW_BUILD_WAYLAND
)

# X11 from the system; Wayland off unless the feature asks for it.
if(VCPKG_TARGET_IS_LINUX AND NOT "wayland" IN_LIST FEATURES)
    list(APPEND FEATURE_OPTIONS -DGLFW_BUILD_WAYLAND=OFF)
endif()
if(VCPKG_TARGET_IS_LINUX)
    list(APPEND FEATURE_OPTIONS -DGLFW_BUILD_X11=ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF
        ${FEATURE_OPTIONS}
    MAYBE_UNUSED_VARIABLES
        GLFW_USE_WAYLAND
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/glfw3)

vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
