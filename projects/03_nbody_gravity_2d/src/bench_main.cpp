// Headless scaling benchmark for the 2D gravity solvers -- the dvec2 twin of
// 02_nbody_gravity's bench_main.cpp. See that file for the reasoning behind
// the timing methodology (untimed warm-up, per-point repetition count, min /
// median / mean / stddev rather than one number, per-call time budget, and
// an accuracy column measured against Direct).
//
// Kept as a separate file rather than shared with 02 because the two
// projects are independent by design: different namespace, different vector
// type, different solver set (2D's far field is a complex-arithmetic FMM).
//
//   03_nbody_scaling_2d --ns 1000,2000,4000 --solvers direct,bh,fmm \
//                       --csv scaling2d.csv

#include "ComplexFmm.hpp"
#include "NBodySystem.hpp"
#include "Quadtree.hpp"
#include "Scenarios.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct SolverSpec {
    const char* key;
    const char* label;
    nbody2d::SolverType type;
};

const SolverSpec kSolvers[] = {
    {"direct", "Direct", nbody2d::SolverType::Direct},
    {"bh", "Barnes-Hut", nbody2d::SolverType::BarnesHut},
    {"fmm", "Complex FMM", nbody2d::SolverType::ComplexFmm},
};

struct Args {
    std::vector<int> ns{1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000};
    std::vector<std::string> solvers{"direct", "bh", "fmm"};
    double targetMs = 2000.0;
    int minReps = 3;
    int maxReps = 25;
    double budgetMs = 30000.0;
    int accuracyMaxN = 8000;
    double theta = -1.0;
    unsigned seed = 1;
    std::string scenario = "cluster";
    std::string csv;
};

std::vector<int> ParseIntList(const char* s) {
    std::vector<int> out;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) out.push_back(static_cast<int>(v));
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

std::vector<std::string> ParseStrList(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--ns") a.ns = ParseIntList(next());
        else if (s == "--solvers") a.solvers = ParseStrList(next());
        else if (s == "--target-ms") a.targetMs = std::atof(next());
        else if (s == "--min-reps") a.minReps = std::atoi(next());
        else if (s == "--max-reps") a.maxReps = std::atoi(next());
        else if (s == "--budget-ms") a.budgetMs = std::atof(next());
        else if (s == "--accuracy-max-n") a.accuracyMaxN = std::atoi(next());
        else if (s == "--theta") a.theta = std::atof(next());
        else if (s == "--seed") a.seed = static_cast<unsigned>(std::atoi(next()));
        else if (s == "--scenario") a.scenario = next();
        else if (s == "--csv") a.csv = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

nbody2d::ScenarioType ScenarioFromName(const std::string& name) {
    if (name == "disk" || name == "rotatingdisk") return nbody2d::ScenarioType::RotatingDisk;
    return nbody2d::ScenarioType::Cluster;
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double Mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double Stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = Mean(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

double RelL2(const std::vector<glm::dvec2>& a, const std::vector<glm::dvec2>& ref) {
    double num = 0.0, den = 0.0;
    const size_t n = std::min(a.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const glm::dvec2 d = a[i] - ref[i];
        num += glm::dot(d, d);
        den += glm::dot(ref[i], ref[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

int ThreadCount() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);
    const nbody2d::ScenarioType scenarioType = ScenarioFromName(a.scenario);
    const int threads = ThreadCount();

    std::FILE* csv = nullptr;
    if (!a.csv.empty()) {
        csv = std::fopen(a.csv.c_str(), "w");
        if (!csv) {
            std::fprintf(stderr, "error: cannot open '%s' for writing\n", a.csv.c_str());
            return 1;
        }
        std::fprintf(csv, "dim,solver,label,n,threads,reps,min_ms,median_ms,mean_ms,std_ms,rel_l2_err\n");
    }

    std::fprintf(stderr, "# 2D gravity solver scaling: scenario=%s seed=%u threads=%d\n",
                 a.scenario.c_str(), a.seed, threads);
    std::fprintf(stderr, "# %-14s %8s %6s %12s %12s %10s %11s\n", "solver", "N", "reps", "median_ms",
                 "min_ms", "std_%", "relL2");

    std::vector<bool> retired(std::size(kSolvers), false);

    for (int n : a.ns) {
        nbody2d::ScenarioParams params;
        params.n = n;
        params.seed = a.seed;
        const nbody2d::ScenarioResult scenario = nbody2d::BuildScenario(scenarioType, params);
        const double theta = (a.theta > 0.0) ? a.theta : scenario.suggestedTheta;

        std::vector<glm::dvec2> reference;
        const bool wantAccuracy = (n <= a.accuracyMaxN);
        if (wantAccuracy) {
            nbody2d::ComputeAccel(nbody2d::SolverType::Direct, scenario.pos, scenario.mass, scenario.G,
                                  scenario.softening, theta, reference);
        }

        for (size_t si = 0; si < std::size(kSolvers); ++si) {
            const SolverSpec& spec = kSolvers[si];
            if (std::find(a.solvers.begin(), a.solvers.end(), spec.key) == a.solvers.end()) continue;
            if (retired[si]) continue;

            std::vector<glm::dvec2> accel;

            const auto w0 = Clock::now();
            nbody2d::ComputeAccel(spec.type, scenario.pos, scenario.mass, scenario.G, scenario.softening,
                                  theta, accel);
            const auto w1 = Clock::now();
            const double warmMs = std::chrono::duration<double, std::milli>(w1 - w0).count();

            if (warmMs > a.budgetMs) {
                std::fprintf(stderr, "# %-14s %8d  retired: one call %.1f ms > budget %.1f ms\n",
                             spec.label, n, warmMs, a.budgetMs);
                retired[si] = true;
                continue;
            }

            int reps = a.minReps;
            if (warmMs > 0.0) {
                reps = static_cast<int>(std::ceil(a.targetMs / warmMs));
                reps = std::max(a.minReps, std::min(a.maxReps, reps));
            }

            std::vector<double> samples;
            samples.reserve(static_cast<size_t>(reps));
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                nbody2d::ComputeAccel(spec.type, scenario.pos, scenario.mass, scenario.G,
                                      scenario.softening, theta, accel);
                const auto t1 = Clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            const double medianMs = Median(samples);
            const double minMs = *std::min_element(samples.begin(), samples.end());
            const double meanMs = Mean(samples);
            const double stdMs = Stddev(samples);
            const double err = (wantAccuracy && spec.type != nbody2d::SolverType::Direct)
                                   ? RelL2(accel, reference)
                                   : 0.0;

            std::fprintf(stderr, "  %-14s %8d %6d %12.3f %12.3f %9.1f%% %11.3e\n", spec.label, n, reps,
                         medianMs, minMs, medianMs > 0.0 ? 100.0 * stdMs / medianMs : 0.0, err);
            std::fflush(stderr);

            if (csv) {
                std::fprintf(csv, "2,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6e\n", spec.key, spec.label, n,
                             threads, reps, minMs, medianMs, meanMs, stdMs, err);
                std::fflush(csv);
            }
        }
    }

    if (csv) std::fclose(csv);
    return 0;
}
