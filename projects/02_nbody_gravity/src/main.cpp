#include "NBodyApp.hpp"
#include "NBodySystem.hpp" // RegisterFmmAdaptersOn

#include "ngrav/DeckSim.hpp"
#include "ngrav/SelfTest.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <glad/glad.h>
#include <stb_image_write.h> // implementation lives in framework/src/SimApp.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string deck;
    std::string out;
    int frames = 0;
    int substeps = 0;
    bool interactive = false;
    bool selftest = false;
    bool dtSelftest = false;
    bool fmmSelftest = false;
    std::string renderCheck;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--frames") a.frames = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--dt-selftest") a.dtSelftest = true;
        else if (s == "--fmm-selftest") a.fmmSelftest = true;
        else if (s == "--render-check") a.renderCheck = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

ngrav::DeckSim<3> MakeSim() {
    return ngrav::DeckSim<3>([](ngrav::System<3>& sys) { nbody::RegisterFmmAdaptersOn(sys); });
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        const bool ok = ngrav::CoreSelfTest3D();
        std::printf("\nselftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.dtSelftest) {
        const bool ok = ngrav::CoreDtSelfTest();
        std::printf("\ndt-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.fmmSelftest) {
        const bool ok = ngrav::CoreFmmSelfTest();
        std::printf("\nfmm-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    // No deck / no explicit mode -> the rich interactive comparison tool.
    if (a.deck.empty() && !a.interactive) {
        nbody::NBodyApp app;
        app.Run();
        return 0;
    }
    if (a.interactive && a.deck.empty()) {
        nbody::NBodyApp app;
        app.Run();
        return 0;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        // Deck-driven interactive still uses the rich NBodyApp; the deck sets
        // the starting scenario/solver via NBodyApp's own future hook (P2+).
        nbody::NBodyApp app;
        app.Run();
        return 0;
    }

    if (!a.renderCheck.empty()) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const int W = 1100, H = 850;
        GLuint fbo = 0, color = 0, depth = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glGenRenderbuffers(1, &depth);
        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "render-check: incomplete FBO\n");
            return 1;
        }
        glViewport(0, 0, W, H);
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        ngrav::DeckSim<3> sim = MakeSim();
        sim.Configure(deck);
        const int warm = deck.GetInt("time.substeps_per_frame", 4) * std::min(deck.GetInt("time.frames", 1), 400);
        if (warm > 1) sim.Step(warm);
        sim.Render(W, H);
        glFinish();

        std::vector<unsigned char> px(static_cast<size_t>(W) * H * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        std::vector<unsigned char> flip(px.size());
        for (int y = 0; y < H; ++y)
            std::memcpy(&flip[(size_t)y * W * 4], &px[(size_t)(H - 1 - y) * W * 4], (size_t)W * 4);
        stbi_write_png(a.renderCheck.c_str(), W, H, 4, flip.data(), W * 4);
        std::printf("render-check -> %s\n", a.renderCheck.c_str());
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  02_nbody_gravity                        (interactive comparison tool)\n"
                     "  02_nbody_gravity --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  02_nbody_gravity --deck <f.toml> --render-check <out.png>\n"
                     "  02_nbody_gravity --selftest\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    ngrav::DeckSim<3> sim = MakeSim();
    sim.Configure(deck);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
