#pragma once

#include "framework/Simulation.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace fw { class Deck; }

namespace fdtd {

// A source term added to E_z at one grid point (soft source): a broadband
// Gaussian pulse or a ramped continuous wave.
struct Source {
    enum Kind { Gaussian, Sine };
    Kind kind = Gaussian;
    int i = 0, j = 0;
    double amplitude = 1.0;
    double f0 = 0.05;       // centre / carrier frequency (cycles per unit time)
    double bandwidth = 0.0; // Gaussian: >0 sets tau; else tau from f0
    double t0 = 0.0;        // Gaussian centre time (0 -> auto, ~4 tau)
    double tau = 0.0;       // filled in at parse time
    double rampCycles = 3.0;// Sine: raised-cosine turn-on over this many periods

    double operator()(double t) const;
};

// 2D transverse-magnetic (TM^z) FDTD on a Yee grid: E_z at the nodes,
// H_x offset half a cell in y, H_y offset half a cell in x; leapfrog in time.
//
//   dE_z/dt = (1/eps)(dH_y/dx - dH_x/dy - sigma E_z)
//   dH_x/dt = -(1/mu) dE_z/dy
//   dH_y/dt =  (1/mu) dE_z/dx
//
// Phase 1: vacuum (eps = mu = 1, sigma = 0), a soft source, and a boundary
// that is either a perfect electric conductor (E_z = 0 on the edge, energy
// conserved) or a first-order Mur absorbing condition. Units: c = 1,
// eps0 = mu0 = 1. Stability (square cells): courant = c dt / dx <= 1/sqrt(2).
class Fdtd2D : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    // Accessors for --selftest.
    int Nx() const { return m_nx; }
    int Ny() const { return m_ny; }
    double Dx() const { return m_dx; }
    double Dt() const { return m_dt; }
    double Time() const { return m_time; }
    long StepCount() const { return m_step; }
    const std::vector<double>& Ez() const { return m_ez; }
    // The exact conserved discrete energy 0.5*sum[eps Ez(n)^2 +
    // mu Hx(n-1/2)Hx(n+1/2) + mu Hy(n-1/2)Hy(n+1/2)], cached each Step just
    // after the H update. Constant to round-off in a closed PEC box.
    double TotalEnergy() const { return m_energy; }
    // The naive 0.5*sum(Ez^2 + Hx^2 + Hy^2) -- ripples with the leapfrog.
    double NaiveEnergy() const;

private:
    std::size_t idx(int i, int j) const {
        return static_cast<std::size_t>(j) * m_nx + i;
    }
    void UpdateH();
    void UpdateE();
    void ApplyMur();
    void InjectSources();

    int m_nx = 200, m_ny = 200;
    double m_dx = 1.0, m_dy = 1.0;
    double m_courant = 0.5;
    double m_dt = 0.0;
    double m_c = 1.0;
    int m_substepsPerFrame = 4;
    long m_totalSteps = 2000;
    std::string m_title = "2D FDTD (TMz)";
    std::string m_boundary = "mur";   // "mur" | "pec"

    std::vector<double> m_ez, m_hx, m_hy;
    std::vector<double> m_ezPrev;      // previous-step E_z, for Mur
    std::vector<double> m_hxPrev, m_hyPrev;  // H(n-1/2), for the exact energy
    std::vector<double> m_ca, m_cb;    // E update coefficients (per cell)
    double m_muInv = 1.0;             // 1/mu, uniform in Phase 1
    double m_energy = 0.0;

    std::vector<Source> m_sources;
    std::vector<std::pair<int, int>> m_probes;

    double m_time = 0.0;
    long m_step = 0;
};

} // namespace fdtd
