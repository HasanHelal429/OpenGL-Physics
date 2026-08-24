#include "Shells.hpp"

#include <algorithm>
#include <array>
#include <sstream>

namespace hf {

std::vector<ShellKey> MadelungOrder(int nMax, int lMax) {
    std::vector<ShellKey> shells;
    for (int n = 1; n <= nMax; ++n) {
        const int lLimit = std::min(n, lMax + 1);
        for (int l = 0; l < lLimit; ++l) {
            shells.emplace_back(n, l);
        }
    }
    std::stable_sort(shells.begin(), shells.end(), [](const ShellKey& a, const ShellKey& b) {
        const int keyA = a.first + a.second;
        const int keyB = b.first + b.second;
        if (keyA != keyB) return keyA < keyB;
        return a.first < b.first;
    });
    return shells;
}

namespace {

Configuration FillMadelung(int Z, int nMax, int lMax) {
    Configuration config;
    int remaining = Z;
    for (const auto& [n, l] : MadelungOrder(nMax, lMax)) {
        if (remaining <= 0) break;
        const int occ = std::min(ShellCapacity(l), remaining);
        config[{n, l}] = occ;
        remaining -= occ;
    }
    return config;
}

std::map<int, Configuration> BuildExceptionsTable() {
    return {
        {24, {{{4, 0}, 1}, {{3, 2}, 5}}},   // Cr: [Ar] 3d5 4s1
        {29, {{{4, 0}, 1}, {{3, 2}, 10}}},  // Cu: [Ar] 3d10 4s1
        {41, {{{5, 0}, 1}, {{4, 2}, 4}}},   // Nb: [Kr] 4d4 5s1
        {42, {{{5, 0}, 1}, {{4, 2}, 5}}},   // Mo: [Kr] 4d5 5s1
        {44, {{{5, 0}, 1}, {{4, 2}, 7}}},   // Ru: [Kr] 4d7 5s1
        {45, {{{5, 0}, 1}, {{4, 2}, 8}}},   // Rh: [Kr] 4d8 5s1
        {46, {{{4, 2}, 10}}},               // Pd: [Kr] 4d10 5s0
        {47, {{{5, 0}, 1}, {{4, 2}, 10}}},  // Ag: [Kr] 4d10 5s1
        {57, {{{6, 0}, 2}, {{5, 2}, 1}}},   // La: [Xe] 5d1 6s2
        {58, {{{6, 0}, 2}, {{4, 3}, 1}, {{5, 2}, 1}}}, // Ce: [Xe] 4f1 5d1 6s2
        {64, {{{6, 0}, 2}, {{4, 3}, 7}, {{5, 2}, 1}}}, // Gd: [Xe] 4f7 5d1 6s2
        {78, {{{6, 0}, 1}, {{4, 3}, 14}, {{5, 2}, 9}}}, // Pt: [Xe] 4f14 5d9 6s1
        {79, {{{6, 0}, 1}, {{4, 3}, 14}, {{5, 2}, 10}}}, // Au: [Xe] 4f14 5d10 6s1
        {89, {{{7, 0}, 2}, {{6, 2}, 1}}},   // Ac: [Rn] 6d1 7s2
        {90, {{{7, 0}, 2}, {{6, 2}, 2}}},   // Th: [Rn] 6d2 7s2
        {91, {{{7, 0}, 2}, {{5, 3}, 2}, {{6, 2}, 1}}}, // Pa: [Rn] 5f2 6d1 7s2
        {92, {{{7, 0}, 2}, {{5, 3}, 3}, {{6, 2}, 1}}}, // U: [Rn] 5f3 6d1 7s2
        {93, {{{7, 0}, 2}, {{5, 3}, 4}, {{6, 2}, 1}}}, // Np: [Rn] 5f4 6d1 7s2
        {96, {{{7, 0}, 2}, {{5, 3}, 7}, {{6, 2}, 1}}}, // Cm: [Rn] 5f7 6d1 7s2
        {103, {{{7, 0}, 2}, {{7, 1}, 1}, {{5, 3}, 14}}}, // Lr: [Rn] 5f14 7s2 7p1 (low confidence)
    };
}

} // namespace

const std::map<int, Configuration>& ExceptionsTable() {
    static const std::map<int, Configuration> table = BuildExceptionsTable();
    return table;
}

Configuration GroundStateConfiguration(int Z, int nMax, int lMax) {
    const auto& exceptions = ExceptionsTable();
    const auto it = exceptions.find(Z);
    if (it == exceptions.end()) {
        return FillMadelung(Z, nMax, lMax);
    }

    int overrideTotal = 0;
    for (const auto& [nl, occ] : it->second) {
        if (occ > 0) overrideTotal += occ;
    }
    const int coreZ = Z - overrideTotal;

    Configuration config = FillMadelung(coreZ, nMax, lMax);
    for (const auto& [nl, occ] : it->second) {
        config.erase(nl);
        if (occ > 0) config[nl] = occ;
    }
    return config;
}

namespace {
constexpr std::array<char, 5> kLLabels = {'s', 'p', 'd', 'f', 'g'};
}

std::string FormatConfiguration(const Configuration& config) {
    std::ostringstream out;
    bool first = true;
    for (const auto& [nl, occ] : config) {
        if (occ <= 0) continue;
        if (!first) out << ' ';
        first = false;
        out << nl.first << kLLabels[static_cast<size_t>(nl.second)] << occ;
    }
    return out.str();
}

const char* ElementSymbol(int Z) {
    static constexpr std::array<const char*, 104> kSymbols = {
        "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr",
    };
    if (Z < 1 || Z > 103) return "?";
    return kSymbols[static_cast<size_t>(Z)];
}

} // namespace hf
