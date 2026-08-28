#include "framework/Deck.hpp"

#include <tomlplusplus/toml.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fw {

namespace {

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Deck: failed to open " + path.string());
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// toml++ value<T>() does not coerce between integer and floating-point nodes,
// so numeric getters try both representations.
template <typename View>
double AsDouble(const View& nv, double fallback) {
    if (auto d = nv.template value<double>()) return *d;
    if (auto i = nv.template value<int64_t>()) return static_cast<double>(*i);
    return fallback;
}

template <typename View>
int64_t AsInt(const View& nv, int64_t fallback) {
    if (auto i = nv.template value<int64_t>()) return *i;
    if (auto d = nv.template value<double>()) return static_cast<int64_t>(*d);
    return fallback;
}

} // namespace

Deck::Deck() = default;
Deck::~Deck() = default;
Deck::Deck(Deck&&) noexcept = default;
Deck& Deck::operator=(Deck&&) noexcept = default;

Deck::Deck(std::shared_ptr<toml::table> root, const toml::table* view)
    : m_root(std::move(root)), m_view(view) {}

Deck Deck::FromString(std::string_view text, std::string sourceName) {
    // The vendored toml++ is built with exceptions enabled: parse() returns a
    // table directly and throws toml::parse_error on failure.
    std::shared_ptr<toml::table> root;
    try {
        root = std::make_shared<toml::table>(toml::parse(text, std::string_view{sourceName}));
    } catch (const toml::parse_error& e) {
        std::ostringstream ss;
        ss << "Deck: parse error in " << sourceName << ": " << e.description() << " at line "
           << e.source().begin.line;
        throw std::runtime_error(ss.str());
    }
    Deck d(root, root.get());
    d.m_sourceText.assign(text);
    d.m_sourceName = std::move(sourceName);
    return d;
}

Deck Deck::FromFile(const std::filesystem::path& path) {
    return FromString(ReadFile(path), path.string());
}

bool Deck::Has(std::string_view path) const {
    return static_cast<bool>(m_view->at_path(path));
}

int Deck::GetInt(std::string_view path, int fallback) const {
    return static_cast<int>(AsInt(m_view->at_path(path), fallback));
}

double Deck::GetDouble(std::string_view path, double fallback) const {
    return AsDouble(m_view->at_path(path), fallback);
}

bool Deck::GetBool(std::string_view path, bool fallback) const {
    return m_view->at_path(path).value<bool>().value_or(fallback);
}

std::string Deck::GetString(std::string_view path, std::string_view fallback) const {
    return m_view->at_path(path).value<std::string>().value_or(std::string{fallback});
}

std::vector<double> Deck::GetDoubleArray(std::string_view path) const {
    std::vector<double> out;
    if (const toml::array* arr = m_view->at_path(path).as_array()) {
        out.reserve(arr->size());
        for (const auto& elem : *arr) {
            out.push_back(AsDouble(elem, 0.0));
        }
    }
    return out;
}

std::vector<std::string> Deck::GetStringArray(std::string_view path) const {
    std::vector<std::string> out;
    if (const toml::array* arr = m_view->at_path(path).as_array()) {
        out.reserve(arr->size());
        for (const auto& elem : *arr) {
            out.push_back(elem.value<std::string>().value_or(std::string{}));
        }
    }
    return out;
}

std::vector<Deck> Deck::GetTables(std::string_view path) const {
    std::vector<Deck> out;
    if (const toml::array* arr = m_view->at_path(path).as_array()) {
        for (const auto& elem : *arr) {
            if (const toml::table* tbl = elem.as_table()) {
                Deck sub(m_root, tbl);
                sub.m_sourceName = m_sourceName;
                out.push_back(std::move(sub));
            }
        }
    }
    return out;
}

} // namespace fw
