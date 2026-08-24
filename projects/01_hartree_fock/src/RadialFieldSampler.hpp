#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace hf {

// This solver's density rho(r) is spherically averaged (isotropic) -- both
// 3D view modes below reduce to exact closed-form geometry in physical
// (Bohr-radius) units; no marching cubes or raymarching needed.

// Samples `count` points distributed according to the radial probability
// 4*pi*r^2*rho(r): inverse-CDF on r (via the already-available cumulative
// integral machinery) plus a uniformly random direction. Positions are in
// physical units (Bohr radii) -- callers apply their own display remap.
std::vector<glm::vec3> SampleRadialParticles(const std::vector<double>& r, const std::vector<double>& rho, int count,
                                              unsigned seed = 1);

// Radii (ascending, physical units) where rho(r) crosses `threshold`. Not
// necessarily a single value -- shell structure means rho(r) can cross a
// given threshold multiple times, which is exactly the concentric-shell
// look an isosurface of this field should show.
std::vector<double> FindIsosurfaceRadii(const std::vector<double>& r, const std::vector<double>& rho, double threshold);

// A reasonable default isosurface threshold: a fraction of the peak density
// within a "visualization window" (small r excluded) rather than the
// global peak, since core density near r~1/Z can be many orders of
// magnitude larger than anything in the valence region and would otherwise
// make every reasonable threshold either miss the core or miss everything else.
double SuggestIsosurfaceThreshold(const std::vector<double>& r, const std::vector<double>& rho, double rWindowMin = 0.05,
                                   double rWindowMax = 10.0);

} // namespace hf
