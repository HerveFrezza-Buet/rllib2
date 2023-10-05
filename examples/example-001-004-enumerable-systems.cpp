#include <cmath>
#include <numbers>
#include <iostream>
#include <tuple>

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
    static constexpr nb_bins {11};
    static continuous::interval to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, -1., 1., nb_bins);}
    static std::size_t from(continuous::interval value) {return rl2::enumerable::utils::digitize::to_index(value, -1., 1., nb_bins);}
  };
  using interval = rl2::enumerable::count<continuous::interval, nb_bins, interval_convertor>;
  
  struct angle_convertor {
    static constexpr nb_bins {24};
    static continuous::angle to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, -1., 1., nb_bins);}
    static std::size_t from(continuous::angle value) {return rl2::enumerable::utils::digitize::to_index(value, -1., 1., nb_bins);}
  };
  using angle = rl2::enumerable::count<continuous::angle, nb_bins, angle_convertor>;

  using plane = rl2::enumerable::pair<interval, interval>;
}


// This is our continuous system

struct circle {
  using state_type       = continuous::angle;
  using command_type     = continuous::angle;
  using observation_type = continuous::plane;
  using report_type      = gdyn::no_report;

  state_type theta = 0;

  void operator=(state_type& theta_init) {theta = theta_init;}
  observation_type operator*() const {double phi =  std::numbers::pi * theta / 180; return {std::cos(phi), std::sin(phi)};}
  report_type operator()(command_type command) {
    theta = command;
    while(theta >= 360) theta -= 360;
    while(theta < 0)    theta += 360;
    return {};
  }
  operator bool() const {return true;}
};

static_assert(specs::transparent_system<circle>);

int main(int argc, char* argv[]) {
  return 0;
}





