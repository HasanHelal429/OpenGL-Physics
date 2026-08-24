#pragma once

#include "Shells.hpp"
#include "framework/Chart2D.hpp"

#include <map>

namespace hf {

// Traditional atomic-spectrum "barcode" look: one vertical stick per
// occupied (n,l) at x=orbital energy, color-coded by l. The x-axis range is
// established from the first snapshot of a run and only ever widens (never
// resets mid-run), so watching the ticks settle into position over
// iterations reads as genuine convergence rather than the axis rescaling
// under them.
class SpectrumView {
public:
    SpectrumView(fw::Font& font, fw::TextRenderer& textRenderer);

    void SetRegion(int x, int y, int width, int height);
    void Reset(); // call when starting a new SCF run

    // Call when a new snapshot arrives, not every frame.
    void UpdateEnergies(const std::map<ShellKey, double>& orbitalEnergies);

    void Render();

private:
    fw::Chart2D m_chart;
    bool m_hasRange = false;
    double m_rangeMin = -1.0;
    double m_rangeMax = -0.001;
};

} // namespace hf
