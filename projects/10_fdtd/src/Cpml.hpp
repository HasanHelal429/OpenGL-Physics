#pragma once

#include <cmath>
#include <vector>

namespace fdtd {

// Convolutional PML (Roden & Gedney) 1D coordinate-stretch profiles for one
// axis. Two staggered sets are needed: one at the field locations that sit on
// integer nodes (E_z), one at the half-node locations (H). Away from the PML
// layer the coefficients are trivial (b = 1, a = 0, kappa = 1) and the CPML
// update reduces exactly to the plain Yee update, so the same code runs
// everywhere with no interior branch.
struct CpmlAxis {
    std::vector<double> bE, aE, kE;   // at integer nodes  (E_z)
    std::vector<double> bH, aH, kH;   // at half nodes      (H)

    // n cells, spacing h, `pml` cells of PML at each end. m is the polynomial
    // grading order, kappaMax the max coordinate stretch, alphaMax the CFS
    // frequency-shift parameter, R0 the target reflection at normal incidence.
    void Build(int n, double h, int pml, double dt,
               double m = 3.0, double kappaMax = 11.0, double alphaMax = 0.15,
               double R0 = 1e-8) {
        bE.assign(n, 1.0); aE.assign(n, 0.0); kE.assign(n, 1.0);
        bH.assign(n, 1.0); aH.assign(n, 0.0); kH.assign(n, 1.0);
        if (pml <= 0) return;

        const double L = pml * h;
        const double sigmaMax = -(m + 1.0) * std::log(R0) / (2.0 * L);  // eta = 1

        auto coeffs = [&](double depth, double& b, double& a, double& k) {
            if (depth <= 0.0) { b = 1.0; a = 0.0; k = 1.0; return; }
            const double x = depth / L;                 // 0 at PML inner edge, 1 at the wall
            const double sigma = sigmaMax * std::pow(x, m);
            k = 1.0 + (kappaMax - 1.0) * std::pow(x, m);
            const double alpha = alphaMax * (1.0 - x);   // linear taper to 0 at the wall
            b = std::exp(-(sigma / k + alpha) * dt);
            const double denom = sigma * k + k * k * alpha;
            a = denom > 1e-30 ? (sigma * (b - 1.0)) / denom : 0.0;
        };

        for (int i = 0; i < n; ++i) {
            // distance of this cell's E node / H node from the nearest wall,
            // measured in physical units, positive inside the PML only
            const double dE_lo = (pml - i) * h;
            const double dE_hi = (i - (n - 1 - pml)) * h;
            const double dH_lo = (pml - (i + 0.5)) * h;
            const double dH_hi = ((i + 0.5) - (n - 1 - pml)) * h;
            coeffs(std::max(dE_lo, dE_hi), bE[i], aE[i], kE[i]);
            coeffs(std::max(dH_lo, dH_hi), bH[i], aH[i], kH[i]);
        }
    }
};

} // namespace fdtd
