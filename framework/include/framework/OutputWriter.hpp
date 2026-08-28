#pragma once

#include "framework/Simulation.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace fw {

class Deck;

// NumPy .npy element type. C8 is complex64 (interleaved re/im float32) -- the
// exact layout of an RG32F GPU buffer readback.
enum class NpyDtype { F4, F8, C8, I4 };

// Writes a batch run to a results directory that Python tooling reads back:
//
//   <dir>/deck.toml          verbatim copy of the input deck
//   <dir>/manifest.json      grid, dt, frame count, field dtypes/shapes, diag names
//   <dir>/diagnostics.csv    header "t,step,<names...>", one row per frame
//   <dir>/frames/psi_0000.npy ... one .npy per field per frame
//
// Usage per frame: BeginFrame(t, step) -> WriteField()/WriteScalar()* -> EndFrame().
// Call Finish() once at the end to emit the manifest.
class OutputWriter {
public:
    OutputWriter(std::filesystem::path outDir, const SimInfo& info, const Deck& deck);

    OutputWriter(const OutputWriter&) = delete;
    OutputWriter& operator=(const OutputWriter&) = delete;

    void BeginFrame(double t, long step);
    // `data` points to ny*nx elements of `dtype`, row-major (index = y*nx + x).
    void WriteField(std::string_view name, const void* data, NpyDtype dtype, int ny, int nx);
    void WriteScalar(std::string_view name, double value);
    void EndFrame();
    void Finish();

    int FramesWritten() const { return m_frameIndex; }

private:
    struct FieldMeta {
        NpyDtype dtype = NpyDtype::F4;
        int ny = 0;
        int nx = 0;
    };

    std::filesystem::path m_dir;
    SimInfo m_info;
    std::ofstream m_csv;
    std::map<std::string, FieldMeta> m_fields; // first-seen shape/dtype per field
    std::map<std::string, double> m_pendingScalars;
    int m_frameIndex = -1;   // incremented by BeginFrame; count = m_frameIndex+1 after EndFrame
    double m_frameT = 0.0;
    long m_frameStep = 0;
    bool m_finished = false;
};

} // namespace fw
