#pragma once

#include <filesystem>

namespace fw {

class Simulation;
class Deck;

struct HeadlessOptions {
    std::filesystem::path outDir;
    int frames = 0;       // 0 -> take from deck [time].frames
    int substeps = 0;     // 0 -> take from deck [time].substeps_per_frame
    bool quiet = false;
};

// Runs `sim` (already Configure()d) as a batch job: writes a frame, advances,
// repeats, then emits the manifest. Returns 0 on success. Needs a current GL
// context (see GLContext::CreateHidden).
int RunHeadless(Simulation& sim, const Deck& deck, const HeadlessOptions& opts);

} // namespace fw
