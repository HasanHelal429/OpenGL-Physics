#pragma once

#include <complex>
#include <vector>

namespace fw { class Deck; }

namespace tdse {

// Uniform periodic grid on [-lx/2, lx/2] x [-ly/2, ly/2], N x N (N a power of 2).
struct Grid {
    int n = 256;
    double lx = 40.0;
    double ly = 40.0;

    double dx() const { return lx / n; }
    double dy() const { return ly / n; }
    double x(int i) const { return -0.5 * lx + (i + 0.5) * dx(); }
    double y(int j) const { return -0.5 * ly + (j + 0.5) * dy(); }
    // FFT angular frequencies (numpy fftfreq * 2pi), matching bin order.
    double kx(int i) const {
        int m = (i <= n / 2) ? i : i - n;
        return 2.0 * 3.14159265358979323846 * m / lx;
    }
    double ky(int j) const {
        int m = (j <= n / 2) ? j : j - n;
        return 2.0 * 3.14159265358979323846 * m / ly;
    }
};

// Real potential V(x,y), row-major (index = j*n + i), summed over [[potential]].
std::vector<float> BuildPotential(const Grid& g, const fw::Deck& deck);

// Non-negative absorbing rate W(x,y) for the border complex-absorbing potential
// (zeros when [boundary].type is "periodic"/"none").
std::vector<float> BuildCap(const Grid& g, const fw::Deck& deck);

// Initial wavefunction from [initial], normalized so sum |psi|^2 dx dy = 1.
std::vector<std::complex<float>> BuildInitial(const Grid& g, const fw::Deck& deck);

} // namespace tdse
