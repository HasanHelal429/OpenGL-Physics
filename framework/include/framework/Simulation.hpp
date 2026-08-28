#pragma once

#include <string>
#include <vector>

namespace fw {

class Deck;
class OutputWriter;

// Static description of a simulation: what it writes and how big the grid is.
// Consumed by OutputWriter (to build the manifest) and by the interactive HUD
// (title text).
struct SimInfo {
    std::string title;
    int gridNx = 0;
    int gridNy = 0;
    double lx = 0.0;
    double ly = 0.0;
    double dt = 0.0;
    int substepsPerFrame = 1;
    // Field arrays written once per frame (e.g. {"psi", "potential"}). A field
    // may opt to only write on frame 0 (static data) -- that is the sim's
    // choice inside Snapshot(), not encoded here.
    std::vector<std::string> frameFields;
    // Scalar diagnostic column names, in order, written after the implicit
    // leading "t" and "step" columns of diagnostics.csv.
    std::vector<std::string> diagnostics;
};

// An interaction event for the interactive view (pan / zoom).
struct ViewInput {
    double dx = 0.0;          // cursor motion since the last event, pixels
    double dy = 0.0;
    double scrollDelta = 0.0; // wheel notches, +up
    bool dragging = false;
    int fbWidth = 1;
    int fbHeight = 1;
};

// The contract every simulator implements. One implementation drives both the
// headless batch runner (Configure -> loop of Step/Snapshot) and the
// interactive host (SimApp: Step on play, Render every frame). Rendering hooks
// are optional -- a pure batch sim can leave them defaulted.
class Simulation {
public:
    virtual ~Simulation() = default;

    // Build initial state from the deck. Requires a current GL context.
    virtual void Configure(const Deck& deck) = 0;
    // Return to the initial condition (same deck, no re-parse).
    virtual void Reset() = 0;
    // Advance the numerics by `substeps` internal steps.
    virtual void Step(int substeps) = 0;
    // Write the current state (fields + scalar diagnostics) for one frame.
    // Not const: may run a GPU reduction / read back into scratch buffers.
    virtual void Snapshot(OutputWriter& writer) = 0;

    virtual SimInfo Info() const = 0;

    // Interactive-only. Default: draw nothing / ignore input.
    virtual void Render(int fbWidth, int fbHeight) { (void)fbWidth; (void)fbHeight; }
    virtual void OnViewInput(const ViewInput& in) { (void)in; }
    // Keys the HUD did not consume (GLFW key + PRESS/REPEAT action).
    virtual void OnKey(int key, int action) { (void)key; (void)action; }
};

} // namespace fw
