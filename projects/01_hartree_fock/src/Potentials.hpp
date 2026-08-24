#pragma once

#include <utility>
#include <vector>

namespace hf {

// Common alpha presets for Slater/X-alpha exchange.
inline constexpr double kAlphaLda = 2.0 / 3.0;     // theoretically exact LDA value
inline constexpr double kAlphaSlater = 1.0;        // classic Slater exchange
inline constexpr double kAlphaSchwarz = 0.7;       // Schwarz-optimized empirical compromise (default)

// Electron-electron Coulomb potential via the shell theorem (O(N), not O(N^2)).
// rho(r) is a number density normalized so integral(rho*4*pi*r**2, dr) = N
// (total electron count).
std::vector<double> HartreePotential(const std::vector<double>& r, const std::vector<double>& rho);

// Local Slater/X-alpha exchange potential: V_x(r) = -3*alpha*(3*rho(r)/(8*pi))**(1/3).
std::vector<double> SlaterExchangePotential(const std::vector<double>& rho, double alpha = kAlphaSchwarz);

// Perdew-Zunger (1981) LDA correlation: energy density per electron eps_c(rs)
// and its potential V_c, via the Wigner-Seitz radius rs = (3/(4*pi*rho))**(1/3).
// Returns {eps_c, V_c}.
std::pair<std::vector<double>, std::vector<double>> Pz81Correlation(const std::vector<double>& rho);

// Genuine Kohn-Sham LDA exchange-correlation potential: exact-LDA Slater
// exchange (alpha=kAlphaLda) plus PZ81 correlation. Returns {V_xc, eps_c}.
std::pair<std::vector<double>, std::vector<double>> LdaXcPotential(const std::vector<double>& rho);

} // namespace hf
