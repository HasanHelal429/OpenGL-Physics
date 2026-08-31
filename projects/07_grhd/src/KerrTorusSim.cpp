#include "KerrTorusSim.hpp"
#include "kernels_kerr2d.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace grhd {

KerrTorusSim::~KerrTorusSim() {
    GLuint bufs[] = {m_cons[0],    m_cons[1],    m_consL[0], m_consL[1], m_stage1, m_stage2,
                     m_stage1L,    m_stage2L,    m_primMain, m_primP,    m_fluxR,  m_fluxRL,
                     m_fluxTh,     m_fluxThL,     m_fixupCount, m_floorCount,
                     m_slopeRMain, m_slopeRP,    m_slopeThMain, m_slopeThP};
    glDeleteBuffers(static_cast<GLsizei>(sizeof(bufs) / sizeof(bufs[0])), bufs);
}

namespace {
// Kerr-Schild (KS) metric + ADM bundle -- mirrors kernels_kerr2d.hpp's
// metricBundleAt() exactly (both derived and independently verified in
// tools/kerr_schild_derive.py and tools/kerr_schild_ref.py; see
// kernels_kerr2d.hpp's header comment for the full derivation story).
struct MetricBundle2D {
    double gtt, gtr, gtphi, grr, grphi, gthth, gphiphi;
    double alpha, betaUpR;
    double gammaUpRr, gammaUpRphi, gammaUpPhiphi, gammaUpThth;
};
MetricBundle2D MetricBundleAt2D(double r, double theta, double M, double a) {
    const double s = std::sin(theta), c = std::cos(theta);
    const double sin2 = s * s, cos2 = c * c;
    const double Sigma = r * r + a * a * cos2;
    const double twoMrOverSigma = 2.0 * M * r / Sigma;
    MetricBundle2D b;
    b.gtt = -(1.0 - twoMrOverSigma);
    b.gtr = twoMrOverSigma;
    b.gtphi = -a * sin2 * twoMrOverSigma;
    b.grr = 1.0 + twoMrOverSigma;
    b.grphi = -a * sin2 * (1.0 + twoMrOverSigma);
    b.gthth = Sigma;
    b.gphiphi = sin2 * (Sigma + a * a * sin2 * (1.0 + twoMrOverSigma));
    b.alpha = std::sqrt(Sigma / (Sigma + 2.0 * M * r));
    b.betaUpR = 2.0 * M * r / (Sigma + 2.0 * M * r);
    const double det2 = b.grr * b.gphiphi - b.grphi * b.grphi;
    b.gammaUpRr = b.gphiphi / det2;
    b.gammaUpRphi = -b.grphi / det2;
    b.gammaUpPhiphi = b.grr / det2;
    b.gammaUpThth = 1.0 / b.gthth;
    return b;
}

// (D,Sr,Stheta,tau,L) from primitives -- v^i is the GENERAL coordinate-
// frame Valencia velocity (KS's spatial metric is not diagonal, unlike
// BL's -- see kernels_kerr2d.hpp's header comment), mirroring
// kernels_kerr2d.hpp's sideState()/kinematics2D() exactly.
void PrimToConsKerr2D(double rho, double vr, double vth, double vphi, double P, double r, double theta,
                       double M, double a, double gamma, glm::vec4& consOut, float& lOut) {
    const MetricBundle2D b = MetricBundleAt2D(r, theta, M, a);
    const double v2 = b.grr * vr * vr + b.gthth * vth * vth + b.gphiphi * vphi * vphi + 2.0 * b.grphi * vr * vphi;
    const double W = 1.0 / std::sqrt(1.0 - v2);
    const double h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    const double ut = W / b.alpha;
    const double ur = W * (vr - b.betaUpR / b.alpha);
    const double uth = W * vth;
    const double uPhiContra = W * vphi; // beta^phi=0 always for KS

    const double betaR = b.grr * b.betaUpR;
    const double betaPhi = b.grphi * b.betaUpR;
    const double urCov = b.grr * ur + b.grphi * uPhiContra + betaR * ut;
    const double uPhiCov = b.grphi * ur + b.gphiphi * uPhiContra + betaPhi * ut;
    const double uthCov = b.gthth * uth;
    const double E = -(b.gtt * ut + b.gtr * ur + b.gtphi * uPhiContra);

    const double sqrtg = b.gthth * std::sin(theta); // sqrt(-g)=Sigma*sin(theta), same identity as BL
    const double D = sqrtg * rho * ut;
    const double Sr = sqrtg * rho * h * ut * urCov;
    const double Sth = sqrtg * rho * h * ut * uthCov;
    const double L = sqrtg * rho * h * ut * uPhiCov;
    const double tau = sqrtg * rho * h * ut * E - sqrtg * P - D;
    consOut = glm::vec4(static_cast<float>(D), static_cast<float>(Sr), static_cast<float>(Sth),
                         static_cast<float>(tau));
    lOut = static_cast<float>(L);
}

// Transforms BL-frame ORTHONORMAL primitives (tools/fishbone_moncrief.py's
// output convention) into KS-frame general Valencia primitives -- r,theta
// unchanged by the BL<->KS coordinate map; only the 4-velocity needs
// transforming. Mirrors tools/kerr_schild_ref.py's bl_to_ks_velocity()
// exactly (verified there against the conserved energy/angular-momentum
// invariants under the coordinate change, to machine precision).
void BlToKsVelocity(double vrBl, double vthBl, double vphiBl, double r, double theta, double M, double a,
                     double& vrKs, double& vthKs, double& vphiKs) {
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double sin2 = std::sin(theta) * std::sin(theta);
    const double cos2 = std::cos(theta) * std::cos(theta);
    const double Sigma = r * r + a * a * cos2;
    const double ABl = (r * r + a * a) * (r * r + a * a) - a * a * Delta * sin2;
    const double alphaBl = std::sqrt(Sigma * Delta / ABl);
    const double gTphiBl = -2.0 * M * a * r * sin2 / Sigma;
    const double gammaRrBl = Sigma / Delta;
    const double gammaThthBl = Sigma;
    const double gammaPhiphiBl = ABl * sin2 / Sigma;
    const double betaPhiUpBl = gTphiBl / gammaPhiphiBl;

    const double W = 1.0 / std::sqrt(1.0 - vrBl * vrBl - vthBl * vthBl - vphiBl * vphiBl);
    const double utBl = W / alphaBl;
    const double urBl = W * vrBl / std::sqrt(gammaRrBl);
    const double uthBl = W * vthBl / std::sqrt(gammaThthBl);
    const double uphiBl = W * vphiBl / std::sqrt(gammaPhiphiBl) - W * betaPhiUpBl / alphaBl;

    const double utKs = utBl + (2.0 * M * r / Delta) * urBl;
    const double urKs = urBl;
    const double uthKs = uthBl;
    const double uphiKs = uphiBl + (a / Delta) * urBl;

    const MetricBundle2D b = MetricBundleAt2D(r, theta, M, a);
    const double WKs = b.alpha * utKs;
    vrKs = urKs / WKs + b.betaUpR / b.alpha;
    vthKs = uthKs / WKs;
    vphiKs = uphiKs / WKs;
}
} // namespace

void KerrTorusSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    m_nr = deck.GetInt("grid.nr", 96);
    m_nth = deck.GetInt("grid.ntheta", 64);
    m_rMin = deck.GetDouble("grid.r_min", 4.0);
    m_rMax = deck.GetDouble("grid.r_max", 30.0);
    m_thetaMin = deck.GetDouble("grid.theta_min", 0.5);
    m_thetaMax = M_PI - m_thetaMin;
    m_dr = (m_rMax - m_rMin) / static_cast<double>(m_nr);
    m_dth = (m_thetaMax - m_thetaMin) / static_cast<double>(m_nth);

    m_M = deck.GetDouble("physics.M", 1.0);
    m_a = deck.GetDouble("physics.a", 0.0);
    m_gamma = deck.GetDouble("physics.gamma", 4.0 / 3.0);

    m_cfl = deck.GetDouble("time.cfl", 0.15);
    m_dt = m_cfl * std::min(m_dr, m_rMin * m_dth); // conservative: smallest physical cell edge, both directions
    m_consToPrimIters = deck.GetInt("solver.cons_to_prim_iters", 40);
    m_rhoFloor = deck.GetDouble("solver.rho_floor", 1e-8);
    m_pFloor = deck.GetDouble("solver.p_floor", 1e-11);
    m_entropyFloor = deck.GetDouble("solver.entropy_floor", 1e-5);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_icFile = deck.GetString("torus.ic_file", "");
    if (m_icFile.empty()) {
        throw std::runtime_error("KerrTorusSim::Configure: deck needs torus.ic_file "
                                  "(see tools/make_fm_torus_ic.py)");
    }

    CreateBuffers();
    UploadInitial();

    m_consToPrim = fw::ComputeShader::FromSource(kernels_kerr2d::ConsToPrim());
    m_computeSlopes = fw::ComputeShader::FromSource(kernels_kerr2d::ComputeSlopes());
    m_fluxesR = fw::ComputeShader::FromSource(kernels_kerr2d::FluxesR());
    m_fluxesTheta = fw::ComputeShader::FromSource(kernels_kerr2d::FluxesTheta());
    m_eulerStep = fw::ComputeShader::FromSource(kernels_kerr2d::EulerStep());
    m_combine = fw::ComputeShader::FromSource(kernels_kerr2d::Combine());
}

void KerrTorusSim::CreateBuffers() {
    const int nCells = m_nr * m_nth;
    const int nFacesR = (m_nr + 1) * m_nth;
    const int nFacesTh = m_nr * (m_nth + 1);

    glCreateBuffers(1, &m_cons[0]);
    glCreateBuffers(1, &m_cons[1]);
    glCreateBuffers(1, &m_consL[0]);
    glCreateBuffers(1, &m_consL[1]);
    glCreateBuffers(1, &m_stage1);
    glCreateBuffers(1, &m_stage2);
    glCreateBuffers(1, &m_stage1L);
    glCreateBuffers(1, &m_stage2L);
    glCreateBuffers(1, &m_primMain);
    glCreateBuffers(1, &m_primP);
    glCreateBuffers(1, &m_fluxR);
    glCreateBuffers(1, &m_fluxRL);
    glCreateBuffers(1, &m_fluxTh);
    glCreateBuffers(1, &m_fluxThL);
    glCreateBuffers(1, &m_fixupCount);
    glCreateBuffers(1, &m_floorCount);
    glCreateBuffers(1, &m_slopeRMain);
    glCreateBuffers(1, &m_slopeRP);
    glCreateBuffers(1, &m_slopeThMain);
    glCreateBuffers(1, &m_slopeThP);

    glNamedBufferData(m_cons[0], nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_cons[1], nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_consL[0], nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_consL[1], nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage1, nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage2, nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage1L, nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage2L, nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_primMain, nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_primP, nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_fluxR, nFacesR * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_fluxRL, nFacesR * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_fluxTh, nFacesTh * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_fluxThL, nFacesTh * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_fixupCount, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_floorCount, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_slopeRMain, nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_slopeRP, nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_slopeThMain, nCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_slopeThP, nCells * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
}

void KerrTorusSim::UploadInitial() {
    std::ifstream f(m_icFile, std::ios::binary);
    if (!f) throw std::runtime_error("KerrTorusSim: cannot open torus.ic_file '" + m_icFile + "'");
    const int nCells = m_nr * m_nth;
    std::vector<float> raw(static_cast<size_t>(nCells) * 5); // rho, v_r, v_theta, v_phi, P per cell
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!f) throw std::runtime_error("KerrTorusSim: '" + m_icFile + "' is smaller than grid.nr*grid.ntheta*5 floats");

    m_initialCons.resize(nCells);
    m_initialConsL.resize(nCells);
    std::vector<glm::vec4> initialPrimMain(nCells);
    std::vector<float> initialPrimP(nCells);
    for (int i = 0; i < m_nr; ++i) {
        for (int j = 0; j < m_nth; ++j) {
            const int idx = i * m_nth + j;
            const double rho = raw[5 * idx + 0];
            // tools/make_fm_torus_ic.py samples the Fishbone-Moncrief
            // solution in BL-frame ORTHONORMAL velocity components (the
            // OLD convention) -- transform to KS-frame general Valencia
            // primitives before use (r,theta unchanged by the BL<->KS
            // coordinate map, only the velocity needs transforming; see
            // BlToKsVelocity's comment and kernels_kerr2d.hpp's header
            // comment for why).
            const double vrBl = raw[5 * idx + 1];
            const double vthBl = raw[5 * idx + 2];
            const double vphiBl = raw[5 * idx + 3];
            const double P = raw[5 * idx + 4];
            const double r = m_rMin + (i + 0.5) * m_dr;
            const double theta = m_thetaMin + (j + 0.5) * m_dth;

            if (rho > 0.0) {
                double vr, vth, vphi;
                BlToKsVelocity(vrBl, vthBl, vphiBl, r, theta, m_M, m_a, vr, vth, vphi);
                glm::vec4 consOut;
                float lOut;
                PrimToConsKerr2D(rho, vr, vth, vphi, P, r, theta, m_M, m_a, m_gamma, consOut, lOut);
                m_initialCons[idx] = consOut;
                m_initialConsL[idx] = lOut;
                initialPrimMain[idx] = glm::vec4(static_cast<float>(rho), static_cast<float>(vr),
                                                  static_cast<float>(vth), static_cast<float>(vphi));
                initialPrimP[idx] = static_cast<float>(P);
            } else {
                // Vacuum floor: a genuine rho=0 has no well-defined W/h: use
                // a tiny floor density at rest instead of a literal zero,
                // matching the "no real neighbors" style floor used
                // elsewhere in this project (e.g. 06_tidal_disruption's
                // zero-density fix) -- avoids div-by-zero in ConsToPrim
                // for cells outside the torus. v^i=0 in the KS Valencia
                // convention already means "comoving with the ZAMO/normal
                // observer" (by construction of that convention, for ANY
                // metric) -- no BL->KS transform needed for this state,
                // it's already the natural "at rest" state in KS terms
                // (physically distinct from BL's v_hat=0 floor, since a KS
                // ZAMO observer is itself falling inward, unlike a BL one
                // -- both are valid, self-consistent floor choices, just
                // not the same physical state).
                const double rhoFloor = 1e-8;
                glm::vec4 consOut;
                float lOut;
                PrimToConsKerr2D(rhoFloor, 0.0, 0.0, 0.0, rhoFloor * 1e-3, r, theta, m_M, m_a, m_gamma, consOut, lOut);
                m_initialCons[idx] = consOut;
                m_initialConsL[idx] = lOut;
                initialPrimMain[idx] = glm::vec4(static_cast<float>(rhoFloor), 0.0f, 0.0f, 0.0f);
                initialPrimP[idx] = static_cast<float>(rhoFloor * 1e-3);
            }
        }
    }
    glNamedBufferSubData(m_cons[0], 0, nCells * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, nCells * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_consL[0], 0, nCells * sizeof(float), m_initialConsL.data());
    glNamedBufferSubData(m_consL[1], 0, nCells * sizeof(float), m_initialConsL.data());
    glNamedBufferSubData(m_primMain, 0, nCells * sizeof(glm::vec4), initialPrimMain.data());
    glNamedBufferSubData(m_primP, 0, nCells * sizeof(float), initialPrimP.data());
    m_cur = 0;
}

void KerrTorusSim::Reset() {
    const int nCells = m_nr * m_nth;
    glNamedBufferSubData(m_cons[0], 0, nCells * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, nCells * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_consL[0], 0, nCells * sizeof(float), m_initialConsL.data());
    glNamedBufferSubData(m_consL[1], 0, nCells * sizeof(float), m_initialConsL.data());
    m_cur = 0;
    m_innerFluxAccum = 0.0;
    m_outerFluxAccum = 0.0;
}

void KerrTorusSim::RunOneRk2Step() {
    const int nCells = m_nr * m_nth;
    const int nFacesR = (m_nr + 1) * m_nth;
    const int nFacesTh = m_nr * (m_nth + 1);
    const GLuint groupsCells = static_cast<GLuint>((nCells + kernels_kerr2d::kWorkgroupSize - 1) / kernels_kerr2d::kWorkgroupSize);
    const GLuint groupsFacesR = static_cast<GLuint>((nFacesR + kernels_kerr2d::kWorkgroupSize - 1) / kernels_kerr2d::kWorkgroupSize);
    const GLuint groupsFacesTh = static_cast<GLuint>((nFacesTh + kernels_kerr2d::kWorkgroupSize - 1) / kernels_kerr2d::kWorkgroupSize);
    const int next = 1 - m_cur;

    auto setCommon = [&](fw::ComputeShader& sh) {
        sh.Use();
        sh.SetInt("uNr", m_nr);
        sh.SetInt("uNth", m_nth);
        sh.SetFloat("uGamma", static_cast<float>(m_gamma));
        sh.SetFloat("uM", static_cast<float>(m_M));
        sh.SetFloat("uA", static_cast<float>(m_a));
        sh.SetFloat("uRmin", static_cast<float>(m_rMin));
        sh.SetFloat("uDr", static_cast<float>(m_dr));
        sh.SetFloat("uThetaMin", static_cast<float>(m_thetaMin));
        sh.SetFloat("uDth", static_cast<float>(m_dth));
    };

    auto consToPrimPass = [&](GLuint consBuf, GLuint consLBuf) {
        unsigned int zero = 0;
        glNamedBufferSubData(m_fixupCount, 0, sizeof(unsigned int), &zero);
        glNamedBufferSubData(m_floorCount, 0, sizeof(unsigned int), &zero);
        setCommon(m_consToPrim);
        m_consToPrim.SetInt("uIters", m_consToPrimIters);
        m_consToPrim.SetFloat("uRhoFloor", static_cast<float>(m_rhoFloor));
        m_consToPrim.SetFloat("uPFloor", static_cast<float>(m_pFloor));
        m_consToPrim.SetFloat("uEntropyFloor", static_cast<float>(m_entropyFloor));
        fw::ComputeShader::BindBuffer(0, consBuf);
        fw::ComputeShader::BindBuffer(6, consLBuf);
        fw::ComputeShader::BindBuffer(4, m_primMain);
        fw::ComputeShader::BindBuffer(10, m_primP);
        fw::ComputeShader::BindBuffer(14, m_fixupCount);
        fw::ComputeShader::BindBuffer(15, m_floorCount);
        m_consToPrim.Dispatch(groupsCells);
        fw::ComputeShader::Barrier();
        unsigned int countThisCall = 0;
        glGetNamedBufferSubData(m_fixupCount, 0, sizeof(unsigned int), &countThisCall);
        m_fixupCountAccum += countThisCall;
        unsigned int floorThisCall = 0;
        glGetNamedBufferSubData(m_floorCount, 0, sizeof(unsigned int), &floorThisCall);
        m_floorCountAccum += floorThisCall;
    };
    // Sums fluxR's D-component (vec4.x) across all theta faces at a given
    // radial face index, weighted by dtheta -- the integral \int F^r_D dtheta
    // at that face, i.e. the instantaneous total mass-flux rate through it.
    auto boundaryRFluxIntegral = [&](int iFace) -> double {
        std::vector<glm::vec4> row(m_nth);
        glGetNamedBufferSubData(m_fluxR, static_cast<GLintptr>(iFace) * m_nth * sizeof(glm::vec4),
                                 m_nth * sizeof(glm::vec4), row.data());
        double sum = 0.0;
        for (int j = 0; j < m_nth; ++j) sum += row[j].x;
        return sum * m_dth;
    };
    struct BoundaryFlux { double inner, outer; };
    auto fluxesPass = [&]() -> BoundaryFlux {
        // Piecewise-linear reconstruction slopes, freshly computed from
        // this stage's primMain/primP -- see kernels_kerr2d.hpp's header
        // comment. Must run before either flux pass below reads them.
        setCommon(m_computeSlopes);
        fw::ComputeShader::BindBuffer(4, m_primMain);
        fw::ComputeShader::BindBuffer(10, m_primP);
        fw::ComputeShader::BindBuffer(16, m_slopeRMain);
        fw::ComputeShader::BindBuffer(17, m_slopeRP);
        fw::ComputeShader::BindBuffer(18, m_slopeThMain);
        fw::ComputeShader::BindBuffer(19, m_slopeThP);
        m_computeSlopes.Dispatch(groupsCells);
        fw::ComputeShader::Barrier();

        setCommon(m_fluxesR);
        fw::ComputeShader::BindBuffer(4, m_primMain);
        fw::ComputeShader::BindBuffer(10, m_primP);
        fw::ComputeShader::BindBuffer(16, m_slopeRMain);
        fw::ComputeShader::BindBuffer(17, m_slopeRP);
        fw::ComputeShader::BindBuffer(5, m_fluxR);
        fw::ComputeShader::BindBuffer(11, m_fluxRL);
        m_fluxesR.Dispatch(groupsFacesR);
        fw::ComputeShader::Barrier();

        setCommon(m_fluxesTheta);
        fw::ComputeShader::BindBuffer(4, m_primMain);
        fw::ComputeShader::BindBuffer(10, m_primP);
        fw::ComputeShader::BindBuffer(18, m_slopeThMain);
        fw::ComputeShader::BindBuffer(19, m_slopeThP);
        fw::ComputeShader::BindBuffer(12, m_fluxTh);
        fw::ComputeShader::BindBuffer(13, m_fluxThL);
        m_fluxesTheta.Dispatch(groupsFacesTh);
        fw::ComputeShader::Barrier();

        return BoundaryFlux{boundaryRFluxIntegral(0), boundaryRFluxIntegral(m_nr)};
    };
    auto eulerPass = [&](GLuint consIn, GLuint consLIn, GLuint consOut, GLuint consLOut) {
        setCommon(m_eulerStep);
        m_eulerStep.SetFloat("uDt", static_cast<float>(m_dt));
        fw::ComputeShader::BindBuffer(0, consIn);
        fw::ComputeShader::BindBuffer(6, consLIn);
        fw::ComputeShader::BindBuffer(4, m_primMain);
        fw::ComputeShader::BindBuffer(10, m_primP);
        fw::ComputeShader::BindBuffer(5, m_fluxR);
        fw::ComputeShader::BindBuffer(11, m_fluxRL);
        fw::ComputeShader::BindBuffer(12, m_fluxTh);
        fw::ComputeShader::BindBuffer(13, m_fluxThL);
        fw::ComputeShader::BindBuffer(1, consOut);
        fw::ComputeShader::BindBuffer(7, consLOut);
        m_eulerStep.Dispatch(groupsCells);
        fw::ComputeShader::Barrier();
    };

    consToPrimPass(m_cons[m_cur], m_consL[m_cur]);
    BoundaryFlux bf1 = fluxesPass();
    eulerPass(m_cons[m_cur], m_consL[m_cur], m_stage1, m_stage1L);

    consToPrimPass(m_stage1, m_stage1L);
    BoundaryFlux bf2 = fluxesPass();
    eulerPass(m_stage1, m_stage1L, m_stage2, m_stage2L);

    // Heun's-method-consistent time integral -- see this class's header
    // comment on m_innerFluxAccum: matches the 0.5*dt*(f(cur)+f(stage1))
    // weighting Combine() below applies to the conserved state itself.
    m_innerFluxAccum += 0.5 * m_dt * (bf1.inner + bf2.inner);
    m_outerFluxAccum += 0.5 * m_dt * (bf1.outer + bf2.outer);

    setCommon(m_combine);
    fw::ComputeShader::BindBuffer(0, m_cons[m_cur]);
    fw::ComputeShader::BindBuffer(1, m_stage2);
    fw::ComputeShader::BindBuffer(2, m_cons[next]);
    fw::ComputeShader::BindBuffer(6, m_consL[m_cur]);
    fw::ComputeShader::BindBuffer(7, m_stage2L);
    fw::ComputeShader::BindBuffer(8, m_consL[next]);
    m_combine.Dispatch(groupsCells);
    fw::ComputeShader::Barrier();

    m_cur = next;
}

void KerrTorusSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        RunOneRk2Step();
    }
}

void KerrTorusSim::ReadBack() {
    const int nCells = m_nr * m_nth;
    m_consCpu.resize(nCells);
    m_consLCpu.resize(nCells);
    m_primMainCpu.resize(nCells);
    m_primPCpu.resize(nCells);
    glGetNamedBufferSubData(m_cons[m_cur], 0, nCells * sizeof(glm::vec4), m_consCpu.data());
    glGetNamedBufferSubData(m_consL[m_cur], 0, nCells * sizeof(float), m_consLCpu.data());
    glGetNamedBufferSubData(m_primMain, 0, nCells * sizeof(glm::vec4), m_primMainCpu.data());
    glGetNamedBufferSubData(m_primP, 0, nCells * sizeof(float), m_primPCpu.data());
}

void KerrTorusSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();
    const int nCells = m_nr * m_nth;

    std::vector<float> rho(nCells), vr(nCells), vth(nCells), vphi(nCells), p(nCells);
    for (int idx = 0; idx < nCells; ++idx) {
        rho[idx] = m_primMainCpu[idx].x;
        vr[idx] = m_primMainCpu[idx].y;
        vth[idx] = m_primMainCpu[idx].z;
        vphi[idx] = m_primMainCpu[idx].w;
        p[idx] = m_primPCpu[idx];
    }

    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, m_nr, m_nth);
    writer.WriteField("v_r", vr.data(), fw::NpyDtype::F4, m_nr, m_nth);
    writer.WriteField("v_theta", vth.data(), fw::NpyDtype::F4, m_nr, m_nth);
    writer.WriteField("v_phi", vphi.data(), fw::NpyDtype::F4, m_nr, m_nth);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, m_nr, m_nth);

    double maxAbsVr = 0.0, maxAbsVth = 0.0, totalMass = 0.0;
    for (int i = 0; i < m_nr; ++i) {
        for (int j = 0; j < m_nth; ++j) {
            const int idx = i * m_nth + j;
            if (rho[idx] > 1e-6) {
                maxAbsVr = std::max<double>(maxAbsVr, std::abs(vr[idx]));
                maxAbsVth = std::max<double>(maxAbsVth, std::abs(vth[idx]));
            }
            totalMass += m_consCpu[idx].x * m_dr * m_dth;
        }
    }
    writer.WriteScalar("max_abs_vr", maxAbsVr);
    writer.WriteScalar("max_abs_vtheta", maxAbsVth);
    writer.WriteScalar("total_mass", totalMass);
    // Diagnostic: how many (cell, ConsToPrim-call) pairs the momentum
    // fixup (kernels_kerr2d.hpp) corrected since the last Snapshot() --
    // see README's Physics section for why this matters.
    writer.WriteScalar("fixup_count", static_cast<double>(m_fixupCountAccum));
    m_fixupCountAccum = 0;
    // Diagnostic: how many cells the vacuum density-floor branch
    // (kernels_kerr2d.hpp) reset since the last Snapshot() -- this branch
    // overwrites D (mass) directly, unlike the momentum fixup above, so
    // it's a separate candidate mechanism for the mass-loss investigation.
    writer.WriteScalar("floor_count", static_cast<double>(m_floorCountAccum));
    m_floorCountAccum = 0;

    // Cumulative since t=0 (NOT reset here, unlike the two counters above)
    // so they're directly comparable to total_mass's own running value --
    // see this class's header comment on m_innerFluxAccum for the exact
    // mass-balance identity this is meant to let the post-processing
    // tooling check.
    writer.WriteScalar("inner_boundary_flux_cum", m_innerFluxAccum);
    writer.WriteScalar("outer_boundary_flux_cum", m_outerFluxAccum);
}

fw::SimInfo KerrTorusSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_nth;
    info.gridNy = m_nr;
    info.lx = m_thetaMax - m_thetaMin;
    info.ly = m_rMax - m_rMin;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "v_r", "v_theta", "v_phi", "p"};
    info.diagnostics = {"max_abs_vr",   "max_abs_vtheta",         "total_mass",
                         "fixup_count", "floor_count",            "inner_boundary_flux_cum",
                         "outer_boundary_flux_cum"};
    return info;
}

} // namespace grhd
