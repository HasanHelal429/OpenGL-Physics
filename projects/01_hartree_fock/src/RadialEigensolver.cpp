#include "RadialEigensolver.hpp"

#include <Eigen/Core>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/GEigsMode.h>
#include <Spectra/Util/SelectionRule.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hf {

RadialMatrices BuildRadialMatrices(const std::vector<double>& r, int l, const std::vector<double>& V, double h) {
    const size_t n = r.size();
    RadialMatrices m;
    m.mainDiagH.resize(n);
    m.offDiagH.resize(n - 1, -1.0 / (h * h));
    m.mDiag.resize(n);

    const double centrifugal = (l + 0.5) * (l + 0.5);
    const double diag0 = 2.0 / (h * h);
    for (size_t i = 0; i < n; ++i) {
        m.mainDiagH[i] = diag0 + centrifugal + 2.0 * r[i] * r[i] * V[i];
        m.mDiag[i] = 2.0 * r[i] * r[i];
    }
    return m;
}

namespace {

// Custom Spectra "A" operator computing y = (H - sigma*M)^-1 * x for our
// tridiagonal H and diagonal M, via an O(N) Thomas-algorithm solve. Faster
// than Spectra's built-in SymShiftInvert (generic dense/sparse LU) because
// it never forms/factorizes anything beyond the tridiagonal coefficients
// themselves — the concrete "optimized vs. Python/ARPACK+SuperLU" win for
// this piece.
class TridiagShiftInvertOp {
public:
    using Scalar = double;

    TridiagShiftInvertOp(const std::vector<double>& mainDiagH, const std::vector<double>& offDiagH,
                          const std::vector<double>& mDiag)
        : m_mainDiagH(mainDiagH), m_offDiagH(offDiagH), m_mDiag(mDiag), m_n(static_cast<Eigen::Index>(mainDiagH.size())),
          m_pivotInv(mainDiagH.size()), m_cPrime(mainDiagH.size() > 0 ? mainDiagH.size() - 1 : 0),
          m_dPrime(mainDiagH.size()) {}

    Eigen::Index rows() const { return m_n; }

    // Factorizes (H - sigma*M) via the Thomas algorithm forward sweep. The
    // resulting pivots/cPrime coefficients depend only on the matrix, not
    // the right-hand side, so they're reused across every perform_op call
    // until the shift changes again.
    void set_shift(const Scalar& sigma) {
        const size_t n = static_cast<size_t>(m_n);
        double b0 = m_mainDiagH[0] - sigma * m_mDiag[0];
        m_pivotInv[0] = 1.0 / b0;
        if (n > 1) {
            m_cPrime[0] = m_offDiagH[0] * m_pivotInv[0];
        }
        for (size_t i = 1; i < n; ++i) {
            const double a = m_offDiagH[i - 1];
            const double b = m_mainDiagH[i] - sigma * m_mDiag[i];
            const double denom = b - a * m_cPrime[i - 1];
            m_pivotInv[i] = 1.0 / denom;
            if (i < n - 1) {
                m_cPrime[i] = m_offDiagH[i] * m_pivotInv[i];
            }
        }
    }

    // y_out = (H - sigma*M)^-1 * x_in, via Thomas algorithm back-substitution
    // using the pivots/cPrime precomputed in set_shift.
    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        const size_t n = static_cast<size_t>(m_n);
        m_dPrime[0] = x_in[0] * m_pivotInv[0];
        for (size_t i = 1; i < n; ++i) {
            const double a = m_offDiagH[i - 1];
            m_dPrime[i] = (x_in[i] - a * m_dPrime[i - 1]) * m_pivotInv[i];
        }
        y_out[n - 1] = m_dPrime[n - 1];
        for (size_t i = n - 1; i-- > 0;) {
            y_out[i] = m_dPrime[i] - m_cPrime[i] * y_out[i + 1];
        }
    }

private:
    const std::vector<double>& m_mainDiagH;
    const std::vector<double>& m_offDiagH;
    const std::vector<double>& m_mDiag;
    Eigen::Index m_n;
    std::vector<double> m_pivotInv;
    std::vector<double> m_cPrime;
    mutable std::vector<double> m_dPrime;
};

// Custom Spectra "B" operator: y = M*x for our diagonal M.
class DiagMatProd {
public:
    using Scalar = double;

    explicit DiagMatProd(const std::vector<double>& diag) : m_diag(diag) {}

    Eigen::Index rows() const { return static_cast<Eigen::Index>(m_diag.size()); }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        for (size_t i = 0; i < m_diag.size(); ++i) {
            y_out[i] = m_diag[i] * x_in[i];
        }
    }

private:
    const std::vector<double>& m_diag;
};

double TrapezoidNormSquared(const std::vector<double>& u, const std::vector<double>& r) {
    double sum = 0.0;
    for (size_t i = 1; i < u.size(); ++i) {
        sum += 0.5 * (u[i] * u[i] + u[i - 1] * u[i - 1]) * (r[i] - r[i - 1]);
    }
    return sum;
}

} // namespace

RadialSolution SolveRadialChannel(const std::vector<double>& r, int l, const std::vector<double>& V, double h,
                                   int nStates, std::optional<double> sigmaOverride) {
    const auto matrices = BuildRadialMatrices(r, l, V, h);
    const size_t n = r.size();

    double sigma;
    if (sigmaOverride) {
        sigma = *sigmaOverride;
    } else {
        // Near r_min, V(r) ~ -Z/r for any physical effective potential.
        const double zEst = -r[0] * V[0];
        sigma = -3.0 * std::max(zEst, 1.0) * std::max(zEst, 1.0);
    }

    for (int attempt = 0; attempt < 4; ++attempt) {
        const int ncvFloor = std::max(4 * nStates + 1, 40) * (1 << attempt);
        const int ncv = std::min(static_cast<int>(n) - 1, ncvFloor);
        const double attemptSigma = sigma * std::pow(1.2, attempt);

        TridiagShiftInvertOp op(matrices.mainDiagH, matrices.offDiagH, matrices.mDiag);
        DiagMatProd bop(matrices.mDiag);

        Spectra::SymGEigsShiftSolver<TridiagShiftInvertOp, DiagMatProd, Spectra::GEigsMode::ShiftInvert> solver(
            op, bop, nStates, ncv, attemptSigma);
        solver.init();
        const int nconv = solver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10, Spectra::SortRule::SmallestAlge);

        if (solver.info() != Spectra::CompInfo::Successful || nconv < nStates) {
            continue;
        }

        const Eigen::VectorXd evalues = solver.eigenvalues();
        const Eigen::MatrixXd evecs = solver.eigenvectors();

        RadialSolution solution;
        solution.energies.assign(evalues.data(), evalues.data() + nStates);
        solution.u.resize(static_cast<size_t>(nStates));
        for (int state = 0; state < nStates; ++state) {
            std::vector<double> u(n);
            for (size_t i = 0; i < n; ++i) {
                u[i] = evecs(static_cast<Eigen::Index>(i), state) * std::sqrt(r[i]);
            }
            const double normSq = TrapezoidNormSquared(u, r);
            const double invNorm = 1.0 / std::sqrt(normSq);
            for (double& value : u) value *= invNorm;
            solution.u[static_cast<size_t>(state)] = std::move(u);
        }
        return solution;
    }

    throw std::runtime_error("SolveRadialChannel: shift-invert Lanczos failed to converge for l=" + std::to_string(l) +
                              ", nStates=" + std::to_string(nStates) + " after retries");
}

} // namespace hf
