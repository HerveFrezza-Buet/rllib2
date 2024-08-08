#pragma once

#include <tuple>

namespace utils {
namespace rbf {

// compute sigma of RBF
inline auto make_bounds(double min, double max, unsigned int nb) {
  return std::make_tuple(min, max, .5 * (max - min) / nb); // min, max, sigma
}

} // namespace rbf
} // namespace utils
