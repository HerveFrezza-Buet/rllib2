#include <cmath>
#include <cstddef>
#include <numbers>
#include <iostream>
#include <tuple>

#include <rllib2.hpp>

// Let us consider a dynamical system whose state is an angle, and
// observation a point position on the trigonometrical circle. The
// command consists in setting the angle.

// We will consider the continuous and discretized case.

namespace continuous {
  using interval = double; // This is supposed to be [-1., 1.]
  using angle    = double; // This is an angle in [0., 360.]
  using plane    = std::pair<interval, interval>;
}

namespace discrete {
  
  struct interval_convertor {
    static constexpr std::size_t nb_bins {11};
    static continuous::interval to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, -1., 1., nb_bins);}
    static std::size_t from(continuous::interval value) {return rl2::enumerable::utils::digitize::to_index(value, -1., 1., nb_bins);}
  };
  using interval = rl2::enumerable::count<continuous::interval, interval_convertor::nb_bins, interval_convertor>;
  
  struct angle_convertor {
    static constexpr std::size_t nb_bins {24};
    static continuous::angle to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, -1., 1., nb_bins);}
    static std::size_t from(continuous::angle value) {return rl2::enumerable::utils::digitize::to_index(value, -1., 1., nb_bins);}
  };
  using angle = rl2::enumerable::count<continuous::angle, angle_convertor::nb_bins, angle_convertor>;

  using plane = rl2::enumerable::pair<interval, interval>;
}


// This is our continuous system

struct circle {
  using state_type       = continuous::angle;
  using command_type     = continuous::angle;
  using observation_type = continuous::plane;
  using report_type      = gdyn::no_report;

  state_type theta = 0;

  void operator=(state_type theta_init) {theta = theta_init;}
  observation_type operator*() const {double phi =  std::numbers::pi * theta / 180; return {std::cos(phi), std::sin(phi)};}
  report_type operator()(command_type command) {
    theta = command;
    while(theta >= 360) theta -= 360;
    while(theta < 0)    theta += 360;
    return {};
  }
  operator bool() const {return true;}
};

struct transparent_circle : public circle {
  using state_type       = circle::state_type;
  using command_type     = circle::command_type;
  using observation_type = circle::observation_type;
  using report_type      = circle::report_type;

  void operator=(state_type theta_init) {this->circle::operator=(theta_init);}
  state_type state() const {return theta;}
};

static_assert(gdyn::specs::system<circle>);
static_assert(gdyn::specs::transparent_system<transparent_circle>);

using discrete_circle             = rl2::enumerable::system<discrete::angle, discrete::plane, discrete::angle, circle>;
using discrete_transparent_circle = rl2::enumerable::system<discrete::angle, discrete::plane, discrete::angle, transparent_circle>;

static_assert(gdyn::specs::system<discrete_circle>);
static_assert(gdyn::specs::transparent_system<discrete_transparent_circle>);

int main(int argc, char* argv[]) {
  return 0;
}





