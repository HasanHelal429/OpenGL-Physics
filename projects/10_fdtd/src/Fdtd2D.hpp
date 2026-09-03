#pragma once

#include "Cpml.hpp"
#include "IncidentWave.hpp"
#include "Materials.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace fw { class Deck; }

namespace fdtd {

// A source term added to E_z at one grid point (soft source): a broadband
// Gaussian pulse or a ramped continuous wave.
struct Source {
    enum Kind { Gaussian, Sine, Tfsf };
    Kind kind = Gaussian;
    int i = 0, j = 0;       // grid point (soft point source; ignored for Tfsf)
    double amplitude = 1.0;
    double f0 = 0.05;       // centre / carrier frequency (cycles per unit time)
    double bandwidth = 0.0; // Gaussian: >0 sets tau; else tau from f0
    double t0 = 0.0;        // Gaussian centre time (0 -> auto, ~4 tau)
    double tau = 0.0;       // filled in at parse time
    double rampCycles = 3.0;// Sine: raised-cosine turn-on over this many periods
    double angleDeg = 90.0; // Tfsf: plane-wave propagation direction from +x

    // The scalar temporal waveform at time t (t < 0 -> 0).
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
    ~Fdtd2D() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    // Force the compute backend on (needs a current GL context); default is
    // read from the deck's solver.backend, else CPU.
    void ForceBackend(bool gpu) { m_useGpu = gpu; }
    bool UsesGpu() const { return m_useGpu; }
    // Pull the GPU field state back into the CPU arrays (Ez()/TotalEnergy()).
    void SyncFromGpu();

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
    void EnforcePec();

    // TFSF plane-wave injection: incident field from the 1D auxiliary grid.
    double IncidentEz(double x, double y) const;
    void IncidentH(double x, double y, double& hx, double& hy) const;
    void TfsfCorrectH();
    void TfsfCorrectE();

    void InitGpu();
    void StepGpu(int substeps);
    void UploadToGpu();

    int m_nx = 200, m_ny = 200;
    double m_dx = 1.0, m_dy = 1.0;
    double m_courant = 0.5;
    double m_dt = 0.0;
    double m_c = 1.0;
    int m_substepsPerFrame = 4;
    long m_totalSteps = 2000;
    bool m_outputH = false;   // also write Hx, Hy frames (near-to-far-field)
    std::string m_title = "2D FDTD (TMz)";
    std::string m_boundary = "mur";   // "mur" | "pec"

    std::vector<double> m_ez, m_hx, m_hy;
    std::vector<double> m_ezPrev;      // previous-step E_z, for Mur
    std::vector<double> m_hxPrev, m_hyPrev;  // H(n-1/2), for the exact energy
    std::vector<double> m_ca, m_cb;    // E update coefficients (per cell)
    std::vector<double> m_muInvCell;   // 1/mu_r per cell (H update)
    double m_energy = 0.0;

    Materials m_mat;
    bool m_hasPec = false;

    // TFSF region: contour indices, propagation angle, causal reference point.
    struct Tfsf {
        bool active = false;
        int i0 = 0, i1 = 0, j0 = 0, j1 = 0;
        double kx = 0.0, ky = -1.0;   // unit propagation direction
        double refX = 0.0, refY = 0.0;
        Source wave;
    } m_tfsf;
    IncidentWave1D m_inc1d;

    // CPML: profiles per axis + the recursive-convolution auxiliary fields.
    // Trivial (b=1, a=0, kappa=1) away from the PML layer, so the same update
    // runs everywhere. Enabled by boundary.type = "cpml".
    int m_pmlCells = 10;
    CpmlAxis m_cpmlX, m_cpmlY;
    std::vector<double> m_psiEzx, m_psiEzy, m_psiHxy, m_psiHyx;

    std::vector<Source> m_sources;
    std::vector<std::pair<int, int>> m_probes;

    double m_time = 0.0;
    long m_step = 0;

    // --- GPU compute backend -------------------------------------------
    bool m_useGpu = false;
    bool m_gpuInit = false;
    bool m_cpuStale = false;        // GPU has advanced past the CPU arrays
    fw::ComputeShader m_kH, m_kE, m_kCopyPrev, m_kInject, m_kMur;
    GLuint m_bEz = 0, m_bHx = 0, m_bHy = 0, m_bCa = 0, m_bCb = 0,
           m_bEzPrev = 0, m_bSrc = 0;
    std::vector<float> m_scratch;   // f32 staging for up/download
    std::vector<glm::vec2> m_srcStage;
};

} // namespace fdtd
