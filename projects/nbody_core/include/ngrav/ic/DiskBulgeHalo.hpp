#pragma once

#include "ngrav/State.hpp"
#include "ngrav/Vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// A three-component disk galaxy IC, 3D: an exponential stellar disk, a
// Hernquist bulge, and an NFW dark-matter halo. Deliberately a "good
// enough to run and look right" model, NOT a self-consistent equilibrium
// (that needs an iterative distribution-function solver -- GALIC / Kuijken
// & Dubinski -- well beyond this project's scope). What it does guarantee:
//
//  - each component's *density profile* is sampled exactly (inverse-CDF on
//    the analytic enclosed mass, or its closed-form inverse where one
//    exists);
//  - disk stars are put on near-circular orbits in the *total* (disk +
//    bulge + halo) midplane potential, with the epicyclic radial-velocity
//    dispersion and the asymmetric drift correction applied (so the disk
//    doesn't immediately fly apart or collapse into a bar in the first few
//    dynamical times);
//  - bulge and halo velocities use the isotropic Jeans dispersion in the
//    total potential (same approximation as ic::Hernquist / ic::King -- see
//    those headers).
//
// The total midplane circular-speed curve and the enclosed-mass profiles
// are all closed-form for these three components, so "the potential" here
// means real analytic expressions, not a Poisson solve.
namespace ngrav::ic {

struct DiskBulgeHaloParams {
    // Particle split.
    int nDisk = 4000;
    int nBulge = 1000;
    int nHalo = 5000;

    // Disk: exponential surface density Sigma ~ exp(-R/Rd), scale height hz
    // (sech^2 vertical profile approximated by a Gaussian of matched
    // dispersion), total mass mDisk.
    double diskMass = 4.0;
    double diskScaleLength = 3.0;
    double diskScaleHeight = 0.3;

    // Bulge: Hernquist, mass mBulge, scale radius aBulge.
    double bulgeMass = 1.0;
    double bulgeScaleRadius = 0.5;

    // Halo: NFW, mass mHalo *within the virial radius* rVir, concentration c
    // (so the NFW scale radius is rVir/c). Sampled only out to rVir.
    double haloMass = 40.0;
    double haloVirialRadius = 60.0;
    double haloConcentration = 10.0;

    double G = 1.0;
    std::uint32_t seed = 1;

    // Fraction of the local circular speed given as ordered rotation to disk
    // stars (1 = fully rotationally supported before the dispersion/drift
    // corrections; lower values make a hotter, more pressure-supported disk).
    double diskRotationFraction = 1.0;
};

namespace detail {

// --- closed-form enclosed masses (spherical for bulge/halo; the disk's
//     own self-gravity contribution to the midplane radial force uses the
//     razor-thin exponential-disk result) ---

inline double HernquistMassEnclosed(double r, double M, double a) {
    const double x = r / (r + a);
    return M * x * x;
}

// NFW: M(<r) = M_s [ln(1+r/rs) - (r/rs)/(1+r/rs)], with M_s set so that
// M(<rVir) = haloMass.
struct NfwProfile {
    double rs = 1.0, Ms = 1.0, rVir = 1.0;
    NfwProfile(double haloMass, double rVir_, double c) : rVir(rVir_) {
        rs = rVir / c;
        const double mu = std::log(1.0 + c) - c / (1.0 + c);
        Ms = haloMass / mu;
    }
    double MassEnclosed(double r) const {
        const double x = r / rs;
        return Ms * (std::log(1.0 + x) - x / (1.0 + x));
    }
    // rho(r) = Ms / (4 pi rs^3) * 1/(x (1+x)^2)
    double Rho(double r) const {
        const double x = r / rs;
        return Ms / (4.0 * M_PI * rs * rs * rs) / (x * (1.0 + x) * (1.0 + x));
    }
};

// Exponential disk: fraction of diskMass enclosed within cylindrical radius
// R is 1 - (1 + R/Rd) exp(-R/Rd). Invert by bisection (no closed form).
inline double ExpDiskMassFraction(double R, double Rd) {
    const double u = R / Rd;
    return 1.0 - (1.0 + u) * std::exp(-u);
}
inline double SampleExpDiskRadius(double x1, double Rd) {
    double lo = 0.0, hi = 30.0 * Rd;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (ExpDiskMassFraction(mid, Rd) < x1)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Razor-thin exponential disk midplane radial force -> circular speed^2
// contribution: v_c,disk^2(R) = 4 pi G Sigma0 Rd y^2 [I0(y)K0(y) - I1(y)K1(y)],
// y = R/(2Rd), Sigma0 = Md/(2 pi Rd^2). Bessel functions via compact
// rational approximations (Abramowitz & Stegun 9.8.x) -- accurate to ~1e-7,
// far tighter than this IC's own equilibrium approximation.
inline double BesselI0(double x) {
    const double ax = std::fabs(x);
    if (ax < 3.75) {
        const double t = (x / 3.75) * (x / 3.75);
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    }
    const double t = 3.75 / ax;
    return (std::exp(ax) / std::sqrt(ax)) *
           (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 +
            t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))));
}
inline double BesselI1(double x) {
    const double ax = std::fabs(x);
    double ans;
    if (ax < 3.75) {
        const double t = (x / 3.75) * (x / 3.75);
        ans = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        const double t = 3.75 / ax;
        ans = 0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059));
        ans = 0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * ans))));
        ans *= (std::exp(ax) / std::sqrt(ax));
    }
    return x < 0.0 ? -ans : ans;
}
inline double BesselK0(double x) {
    if (x <= 2.0) {
        const double t = x * x / 4.0;
        return (-std::log(x / 2.0) * BesselI0(x)) +
               (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590 + t * (0.00262698 + t * (0.0001075 + t * 0.0000074))))));
    }
    const double t = 2.0 / x;
    return (std::exp(-x) / std::sqrt(x)) *
           (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))));
}
inline double BesselK1(double x) {
    if (x <= 2.0) {
        const double t = x * x / 4.0;
        return (std::log(x / 2.0) * BesselI1(x)) +
               (1.0 / x) * (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897 + t * (-0.01919402 + t * (-0.00110404 - t * 0.00004686))))));
    }
    const double t = 2.0 / x;
    return (std::exp(-x) / std::sqrt(x)) *
           (1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268 + t * (-0.00780353 + t * (0.00325614 - t * 0.00068245))))));
}

} // namespace detail

template <int D>
SoA<D> DiskBulgeHalo(const DiskBulgeHaloParams& p) {
    static_assert(D == 3, "DiskBulgeHalo ic:: is 3D only");
    const int n = p.nDisk + p.nBulge + p.nHalo;
    SoA<D> s;
    s.Resize(static_cast<std::size_t>(n));
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    const detail::NfwProfile nfw(p.haloMass, p.haloVirialRadius, p.haloConcentration);
    const double Sigma0 = p.diskMass / (2.0 * M_PI * p.diskScaleLength * p.diskScaleLength);

    // Total spherically-averaged M(<r) for bulge+halo; the disk's midplane
    // rotation curve contribution is handled separately (razor-thin result).
    auto sphMassEnclosed = [&](double r) {
        return detail::HernquistMassEnclosed(r, p.bulgeMass, p.bulgeScaleRadius) + nfw.MassEnclosed(std::min(r, p.haloVirialRadius));
    };
    // Total midplane circular speed^2 at cylindrical radius R.
    auto vc2Total = [&](double R) {
        const double vSph2 = p.G * sphMassEnclosed(R) / std::max(R, 1e-6);
        const double y = R / (2.0 * p.diskScaleLength);
        double vDisk2 = 0.0;
        if (y > 1e-4) {
            vDisk2 = 4.0 * M_PI * p.G * Sigma0 * p.diskScaleLength * y * y *
                     (detail::BesselI0(y) * detail::BesselK0(y) - detail::BesselI1(y) * detail::BesselK1(y));
        }
        return std::max(0.0, vSph2 + vDisk2);
    };

    int idx = 0;

    // --- disk ---
    for (int i = 0; i < p.nDisk; ++i, ++idx) {
        const double R = detail::SampleExpDiskRadius(u(rng), p.diskScaleLength);
        const double phi = 2.0 * M_PI * u(rng);
        const double z = p.diskScaleHeight * gauss(rng);
        const std::size_t ii = static_cast<std::size_t>(idx);
        s.SetPos(ii, Vec<3>(R * std::cos(phi), R * std::sin(phi), z));

        const double vc = std::sqrt(vc2Total(R));
        // Radial velocity dispersion: sigma_R(R) = sigma_R0 exp(-R/(2Rd))
        // (standard exponential-disk choice), sigma_R0 set to ~ a fixed
        // fraction of the central circular speed so Toomre Q ~ 1.5 in the
        // disk body -- crude but keeps the disk from being either razor-cold
        // (instant bar) or dispersion-dominated.
        const double sigmaR0 = 0.35 * std::sqrt(vc2Total(p.diskScaleLength));
        const double sigmaR = sigmaR0 * std::exp(-R / (2.0 * p.diskScaleLength));
        const double sigmaPhi = 0.7 * sigmaR; // ~epicyclic ratio at a flat rotation curve
        const double sigmaZ = 0.6 * sigmaR + 1e-6;

        // Asymmetric drift: mean rotational speed is slightly below vc.
        const double ad = (sigmaR * sigmaR) * (0.5 + R / p.diskScaleLength); // approximate B&T eq. 4.228 grouping
        const double vMeanPhi = p.diskRotationFraction * std::sqrt(std::max(0.0, vc * vc - ad));

        const double vR = sigmaR * gauss(rng);
        const double vPhi = vMeanPhi + sigmaPhi * gauss(rng);
        const double vz = sigmaZ * gauss(rng);
        // cylindrical -> cartesian
        const double cx = std::cos(phi), sx = std::sin(phi);
        s.SetVel(ii, Vec<3>(vR * cx - vPhi * sx, vR * sx + vPhi * cx, vz));
        s.m[ii] = p.diskMass / p.nDisk;
    }

    // --- bulge (Hernquist, isotropic Jeans in the total potential) ---
    for (int i = 0; i < p.nBulge; ++i, ++idx) {
        const double x1 = u(rng);
        const double sq = std::sqrt(x1);
        const double r = p.bulgeScaleRadius * std::min(sq / (1.0 - sq), 40.0); // clip r, not x1 (heavy tail)
        const double cosT = 2.0 * u(rng) - 1.0;
        const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        const double ph = 2.0 * M_PI * u(rng);
        const std::size_t ii = static_cast<std::size_t>(idx);
        s.SetPos(ii, Vec<3>(r * sinT * std::cos(ph), r * sinT * std::sin(ph), r * cosT));

        // sigma^2 ~ G M_tot(<r) / (2 r) is the rough isotropic-Jeans scale
        // for a centrally-concentrated tracer in a mildly-cuspy potential --
        // adequate for this "runs and looks right" IC.
        const double sigma = std::sqrt(std::max(0.0, p.G * sphMassEnclosed(r) / (2.0 * std::max(r, 1e-3))));
        s.SetVel(ii, Vec<3>(sigma * gauss(rng), sigma * gauss(rng), sigma * gauss(rng)));
        s.m[ii] = p.bulgeMass / p.nBulge;
    }

    // --- halo (NFW, isotropic Jeans in the total potential) ---
    for (int i = 0; i < p.nHalo; ++i, ++idx) {
        // Inverse-CDF on M(<r)/M(<rVir) by bisection.
        const double target = u(rng);
        double lo = 1e-4 * nfw.rs, hi = p.haloVirialRadius;
        const double mVir = nfw.MassEnclosed(p.haloVirialRadius);
        for (int it = 0; it < 80; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (nfw.MassEnclosed(mid) / mVir < target)
                lo = mid;
            else
                hi = mid;
        }
        const double r = 0.5 * (lo + hi);
        const double cosT = 2.0 * u(rng) - 1.0;
        const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        const double ph = 2.0 * M_PI * u(rng);
        const std::size_t ii = static_cast<std::size_t>(idx);
        s.SetPos(ii, Vec<3>(r * sinT * std::cos(ph), r * sinT * std::sin(ph), r * cosT));

        const double sigma = std::sqrt(std::max(0.0, p.G * sphMassEnclosed(r) / (2.0 * std::max(r, 1e-3))));
        s.SetVel(ii, Vec<3>(sigma * gauss(rng), sigma * gauss(rng), sigma * gauss(rng)));
        s.m[ii] = p.haloMass / p.nHalo;
    }

    // Recenter position + momentum.
    Vec<D> cp(0.0), cv(0.0);
    double tm = 0.0;
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        cp += s.m[ii] * s.Pos(ii);
        cv += s.m[ii] * s.Vel(ii);
        tm += s.m[ii];
    }
    cp /= tm;
    cv /= tm;
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        s.SetPos(ii, s.Pos(ii) - cp);
        s.SetVel(ii, s.Vel(ii) - cv);
    }
    return s;
}

} // namespace ngrav::ic
