#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hf {

using ShellKey = std::pair<int, int>;      // (n, l)
using Configuration = std::map<ShellKey, int>; // occupation per shell, sorted by (n, l)

// (n, l) shells in Madelung fill order: increasing (n+l), then increasing n.
std::vector<ShellKey> MadelungOrder(int nMax = 8, int lMax = 3);

// Max electron occupancy 2*(2l+1) for a shell of angular momentum l.
inline int ShellCapacity(int l) { return 2 * (2 * l + 1); }

// Real-world ground-state configurations deviate from naive Madelung filling
// for these elements (subtle exchange/correlation effects a configuration-
// averaged, non-relativistic Xalpha/LDA mean field can't derive from first
// principles). Each entry gives only the *valence* shells that differ from
// the Madelung-predicted pattern; the noble-gas core underneath is still
// ordinary Madelung filling. Checked against NIST ASD. Z=103 (Lr) is
// lower-confidence: sensitive to relativistic effects outside this solver's
// scope.
const std::map<int, Configuration>& ExceptionsTable();

// Fixed (n, l) -> occupation for the ground-state configuration of atomic
// number Z. Occupations are held fixed for a given Z throughout an SCF run.
Configuration GroundStateConfiguration(int Z, int nMax = 8, int lMax = 3);

// Human-readable string like "1s2 2s2 2p6 3s2 3p6 3d5 4s1".
std::string FormatConfiguration(const Configuration& config);

// Element symbol for atomic number Z (1..103), e.g. ElementSymbol(26) == "Fe".
const char* ElementSymbol(int Z);

} // namespace hf
