#include "ngrav/SelfTest.hpp"

#include <cstdio>

// Standalone runner for the core numerics validation (no GL). Also reachable
// as `02_nbody_gravity --selftest` / `03_nbody_gravity_2d --selftest`.
int main() {
    const bool ok3 = ngrav::CoreSelfTest3D();
    const bool ok2 = ngrav::CoreSelfTest2D();
    const bool okdt = ngrav::CoreDtSelfTest();
    const bool ok = ok3 && ok2 && okdt;
    std::printf("\nnbody_core selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
