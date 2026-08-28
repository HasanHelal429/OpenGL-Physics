#include "framework/HeadlessRunner.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"
#include "framework/Simulation.hpp"

#include <cstdio>

namespace fw {

int RunHeadless(Simulation& sim, const Deck& deck, const HeadlessOptions& opts) {
    const SimInfo info = sim.Info();

    int frames = opts.frames > 0 ? opts.frames : deck.GetInt("time.frames", 200);
    int substeps = opts.substeps > 0 ? opts.substeps : info.substepsPerFrame;
    if (frames < 1) frames = 1;
    if (substeps < 1) substeps = 1;

    OutputWriter writer(opts.outDir, info, deck);

    for (int f = 0; f < frames; ++f) {
        const long step = static_cast<long>(f) * substeps;
        const double t = static_cast<double>(step) * info.dt;

        writer.BeginFrame(t, step);
        sim.Snapshot(writer);
        writer.EndFrame();

        if (!opts.quiet) {
            std::printf("\rframe %d/%d  t=%.4f   ", f + 1, frames, t);
            std::fflush(stdout);
        }
        if (f + 1 < frames) sim.Step(substeps);
    }

    writer.Finish();
    if (!opts.quiet) {
        std::printf("\ndone -> %s\n", opts.outDir.string().c_str());
    }
    return 0;
}

} // namespace fw
