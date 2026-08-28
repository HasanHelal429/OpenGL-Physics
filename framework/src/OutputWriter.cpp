#include "framework/OutputWriter.hpp"

#include "framework/Deck.hpp"

#include <array>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fw {

namespace {

const char* DtypeDescr(NpyDtype d) {
    switch (d) {
        case NpyDtype::F4: return "<f4";
        case NpyDtype::F8: return "<f8";
        case NpyDtype::C8: return "<c8";
        case NpyDtype::I4: return "<i4";
    }
    return "<f4";
}

size_t DtypeSize(NpyDtype d) {
    switch (d) {
        case NpyDtype::F4: return 4;
        case NpyDtype::F8: return 8;
        case NpyDtype::C8: return 8;
        case NpyDtype::I4: return 4;
    }
    return 4;
}

// NumPy .npy v1.0: 6-byte magic, 2 version bytes, 2-byte LE header length, then
// an ASCII dict padded with spaces so the whole preamble is a multiple of 64,
// terminated by '\n'.
void WriteNpy(const std::filesystem::path& path, const void* data, NpyDtype dtype, int ny, int nx) {
    std::ostringstream dict;
    dict << "{'descr': '" << DtypeDescr(dtype) << "', 'fortran_order': False, 'shape': (";
    if (ny > 0) {
        dict << ny << ", " << nx;
    } else {
        dict << nx << ",";  // 1-D
    }
    dict << "), }";
    std::string header = dict.str();

    const size_t preambleUnpadded = 10 + header.size() + 1; // magic+ver+len + dict + '\n'
    const size_t padded = (preambleUnpadded + 63) & ~size_t(63);
    header.append(padded - preambleUnpadded, ' ');
    header.push_back('\n');
    const uint16_t headerLen = static_cast<uint16_t>(header.size());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("OutputWriter: cannot open " + path.string());
    const std::array<char, 8> magic{'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    f.write(magic.data(), magic.size());
    const char lo = static_cast<char>(headerLen & 0xFF);
    const char hi = static_cast<char>((headerLen >> 8) & 0xFF);
    f.put(lo).put(hi);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));

    const size_t count = static_cast<size_t>(ny > 0 ? ny : 1) * static_cast<size_t>(nx);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * DtypeSize(dtype)));
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

std::string FrameTag(int idx) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%04d", idx);
    return buf;
}

} // namespace

OutputWriter::OutputWriter(std::filesystem::path outDir, const SimInfo& info, const Deck& deck)
    : m_dir(std::move(outDir)), m_info(info) {
    std::filesystem::create_directories(m_dir / "frames");

    {
        std::ofstream deckCopy(m_dir / "deck.toml", std::ios::binary);
        deckCopy << deck.SourceText();
    }

    m_csv.open(m_dir / "diagnostics.csv", std::ios::binary);
    if (!m_csv) throw std::runtime_error("OutputWriter: cannot open diagnostics.csv");
    m_csv << "t,step";
    for (const auto& name : m_info.diagnostics) m_csv << ',' << name;
    m_csv << '\n';
}

void OutputWriter::BeginFrame(double t, long step) {
    ++m_frameIndex;
    m_frameT = t;
    m_frameStep = step;
    m_pendingScalars.clear();
}

void OutputWriter::WriteField(std::string_view name, const void* data, NpyDtype dtype, int ny, int nx) {
    const std::string key(name);
    if (!m_fields.count(key)) m_fields[key] = FieldMeta{dtype, ny, nx};
    WriteNpy(m_dir / "frames" / (key + "_" + FrameTag(m_frameIndex) + ".npy"), data, dtype, ny, nx);
}

void OutputWriter::WriteScalar(std::string_view name, double value) {
    m_pendingScalars[std::string(name)] = value;
}

void OutputWriter::EndFrame() {
    m_csv << std::setprecision(10) << m_frameT << ',' << m_frameStep;
    for (const auto& name : m_info.diagnostics) {
        auto it = m_pendingScalars.find(name);
        m_csv << ',';
        if (it != m_pendingScalars.end()) m_csv << it->second;
        else m_csv << "nan";
    }
    m_csv << '\n';
    m_csv.flush();
}

void OutputWriter::Finish() {
    if (m_finished) return;
    m_finished = true;
    m_csv.close();

    const int frames = m_frameIndex + 1;
    std::ofstream j(m_dir / "manifest.json", std::ios::binary);
    j << "{\n";
    j << "  \"title\": \"" << JsonEscape(m_info.title) << "\",\n";
    j << "  \"grid\": { \"nx\": " << m_info.gridNx << ", \"ny\": " << m_info.gridNy
      << ", \"lx\": " << m_info.lx << ", \"ly\": " << m_info.ly << " },\n";
    j << "  \"dt\": " << m_info.dt << ",\n";
    j << "  \"substeps_per_frame\": " << m_info.substepsPerFrame << ",\n";
    j << "  \"frames\": " << frames << ",\n";
    j << "  \"deck_file\": \"deck.toml\",\n";

    j << "  \"fields\": {";
    bool first = true;
    for (const auto& [name, meta] : m_fields) {
        j << (first ? "\n" : ",\n") << "    \"" << JsonEscape(name) << "\": { \"dtype\": \""
          << DtypeDescr(meta.dtype) << "\", \"shape\": [" << meta.ny << ", " << meta.nx << "] }";
        first = false;
    }
    j << (first ? "" : "\n  ") << "},\n";

    j << "  \"diagnostics\": [";
    for (size_t i = 0; i < m_info.diagnostics.size(); ++i) {
        j << (i ? ", " : "") << '"' << JsonEscape(m_info.diagnostics[i]) << '"';
    }
    j << "]\n";
    j << "}\n";
}

} // namespace fw
