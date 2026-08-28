#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace toml { inline namespace v3 { class table; } }

namespace fw {

// A parsed TOML "input deck": the single source of every knob a simulation
// needs, so batch and interactive runs are reproducible from one file.
//
// Access is by dotted path ("grid.n", "time.dt"). Missing keys return the
// supplied default rather than throwing, so decks can stay terse. Arrays of
// tables ([[potential]]) are read with GetTables().
class Deck {
public:
    static Deck FromFile(const std::filesystem::path& path);
    static Deck FromString(std::string_view toml, std::string sourceName = "<string>");

    Deck(Deck&&) noexcept;
    Deck& operator=(Deck&&) noexcept;
    Deck(const Deck&) = delete;
    Deck& operator=(const Deck&) = delete;
    ~Deck();

    bool Has(std::string_view path) const;

    int GetInt(std::string_view path, int fallback) const;
    double GetDouble(std::string_view path, double fallback) const;
    bool GetBool(std::string_view path, bool fallback) const;
    std::string GetString(std::string_view path, std::string_view fallback) const;
    std::vector<double> GetDoubleArray(std::string_view path) const;
    std::vector<std::string> GetStringArray(std::string_view path) const;

    // Each element of an array-of-tables, wrapped as its own Deck (paths are
    // then relative to that sub-table). Empty if the path is absent.
    std::vector<Deck> GetTables(std::string_view path) const;

    // Verbatim source text (for OutputWriter to copy into the results dir).
    const std::string& SourceText() const { return m_sourceText; }
    const std::string& SourceName() const { return m_sourceName; }

private:
    Deck();
    explicit Deck(std::shared_ptr<toml::table> root, const toml::table* view);

    std::shared_ptr<toml::table> m_root;   // keeps the tree alive
    const toml::table* m_view = nullptr;   // sub-table this Deck resolves against
    std::string m_sourceText;
    std::string m_sourceName;
};

} // namespace fw
