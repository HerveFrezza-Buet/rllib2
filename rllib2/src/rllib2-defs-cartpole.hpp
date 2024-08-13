#pragma once

#include <random>
#include <gdyn.hpp>
#include <rllib2.hpp>

namespace rl2 {
namespace problem {
namespace defs {
namespace cartpole {

  // This is the state space.
  using S = gdyn::problem::cartpole::state;

  // The action space is a enumerable version of gdyn::problem::cartpole::direction.
  struct A_convertor {
    static constexpr std::size_t size() {return 2;}
    static gdyn::problem::cartpole::direction to(std::size_t index) {
      if(index ==  0) return gdyn::problem::cartpole::direction::Left;
      else            return gdyn::problem::cartpole::direction::Right;
    }
    static std::size_t from(const gdyn::problem::cartpole::direction& d)  {
      return static_cast<std::size_t>(d);
    }
  };
  using A = rl2::enumerable::set<gdyn::problem::cartpole::direction, A_convertor::size(), A_convertor>;


  // Let us set up the Gaussian RBF for states.  S is a struct with 4
  // **contiguous** attributes with type double.

  struct wrapper {
    constexpr static std::size_t dim = 4;
    const double* start;
    const double* sentinel;
    wrapper(const S& data)
      : start    (reinterpret_cast<const double*>(&data)),
        sentinel (reinterpret_cast<const double*>(&data) + dim) {}
    auto begin() const {return start;}
    auto end()   const {return sentinel;}
  };

  using mu_type = rl2::nuplet::from<double, wrapper::dim>;
  using rbf = rl2::functional::gaussian<mu_type, S, wrapper>;

} // namespace cartpole
} // namespace defs
} // namespace problem
} // namespace rl2
