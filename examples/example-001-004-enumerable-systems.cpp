#include <cmath>
#include <cstddef>
#include <numbers>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <ranges>
	

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
    static continuous::angle to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, 0., 360., nb_bins);}
    static std::size_t from(continuous::angle value) {return rl2::enumerable::utils::digitize::to_index(value, 0., 360., nb_bins);}
  };
  using angle = rl2::enumerable::count<continuous::angle, angle_convertor::nb_bins, angle_convertor>;

  using plane = rl2::enumerable::pair<interval, interval>;
}


// This is our continuous systems

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
    while(theta <    0) theta += 360;
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

static_assert(gdyn::concepts::system<circle>);
static_assert(gdyn::concepts::transparent_system<transparent_circle>);


// This is our discrete systems

using discrete_circle             = rl2::enumerable::system<discrete::angle, discrete::plane, discrete::angle, circle>;
using discrete_transparent_circle = rl2::enumerable::system<discrete::angle, discrete::plane, discrete::angle, transparent_circle>;

static_assert(gdyn::concepts::system<discrete_circle>);
static_assert(gdyn::concepts::transparent_system<discrete_transparent_circle>);

#define WIDTH 8
int main(int argc, char* argv[]) {

  transparent_circle system;
  discrete_transparent_circle dsystem(system);

  system = continuous::angle(0);

  std::cout <<  "| " << std::setw(WIDTH) << "state"
	    << " | " << std::setw(WIDTH) << "dstate"
	    << " | " << std::setw(WIDTH) << "sta_idx"
	    << " | " << std::setw(WIDTH) << "x"
	    << " | " << std::setw(WIDTH) << "y"
	    << " | " << std::setw(WIDTH) << "dx"
	    << " | " << std::setw(WIDTH) << "dy"
	    << " | " << std::setw(WIDTH) << "obs_idx"
	    << " |" << std::endl;
  
  std::string chunk = std::string("+") + std::string(WIDTH+2, '-');
  std::string bar;
  for(int i = 0; i < 8; ++i, bar += chunk);
  bar += std::string("+");
  std::cout << bar << std::endl;
  
  for(auto cmd
	: std::views::iota(0)
	| std::views::transform([](auto x) -> double {return x;})
	| std::views::take(360)) {
    system(cmd);
    auto [x,   y] = *system;
    auto [dx, dy] = static_cast<discrete_transparent_circle::observation_type::base_type>(*dsystem);
    auto obs_idx  = static_cast<std::size_t>(*dsystem);
    auto state    = system.state();
    auto dstate   = static_cast<discrete_transparent_circle::state_type::base_type>(dsystem.state());
    auto sta_idx  = static_cast<std::size_t>(dsystem.state());
    std::cout << std::fixed << std::setprecision(4)
	      <<  "| " << std::setw(WIDTH) << state
	      << " | " << std::setw(WIDTH) << dstate
	      << " | " << std::setw(WIDTH) << sta_idx
	      << " | " << std::setw(WIDTH) << x
	      << " | " << std::setw(WIDTH) << y
	      << " | " << std::setw(WIDTH) << dx
	      << " | " << std::setw(WIDTH) << dy
	      << " | " << std::setw(WIDTH) << obs_idx
	      << " |" << std::endl;
      
  }
  
  std::cout << bar << std::endl;
  
  
  return 0;
}





