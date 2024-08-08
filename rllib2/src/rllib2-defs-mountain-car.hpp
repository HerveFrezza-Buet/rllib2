#pragma once

#include <gdyn.hpp>
#include <random>
#include <rllib2.hpp>

namespace rl2 {
namespace problem {
namespace defs {
namespace mountain_car {

// This is the state space.
using S = gdyn::problem::mountain_car::state;

/** The action space is a enumerable version of gdyn::problem::mountain_car::acceleration */
struct A_convertor {
  static constexpr std::size_t size() { return 3; }
  static gdyn::problem::mountain_car::acceleration to(std::size_t index) {
    if (index == 0)
      return gdyn::problem::mountain_car::acceleration::Left;
    else if (index == 1)
      return gdyn::problem::mountain_car::acceleration::None;
    else
      return gdyn::problem::mountain_car::acceleration::Right;
  }
  static std::size_t from(const gdyn::problem::mountain_car::acceleration &d) {
    return static_cast<std::size_t>(d);
  }
};
using A = rl2::enumerable::set<gdyn::problem::mountain_car::acceleration,
                               A_convertor::size(), A_convertor>;

/** Let us set up the Gaussian RBF for states.
 * Leverage on the fact that S is a struct with 2 **contiguous** attributes with type double
 * to provide an iterator
 */
struct wrapper {
  constexpr static std::size_t dim = 2;
  const double *start;
  const double *sentinel;
  wrapper(const S &data)
      : start(reinterpret_cast<const double *>(&data)),
        sentinel(reinterpret_cast<const double *>(&data) + dim) {}
  auto begin() const { return start; }
  auto end() const { return sentinel; }
};

using mu_type = rl2::nuplet::from<double, wrapper::dim>;
using rbf = rl2::functional::gaussian<mu_type, S, wrapper>;

} // namespace moutain_car
} // namespace defs
} // namespace problem
} // namespace rl2
