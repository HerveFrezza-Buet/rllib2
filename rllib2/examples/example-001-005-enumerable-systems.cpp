#include <cmath>
#include <cstddef>
#include <numbers>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <ranges>
	

#include <rllib2.hpp>

#define ALPHA .5

// Let us consider a dynamical system made of 
// single scalar 'x' as a state. The command 'a' is a scalar as well
// (both are positive). At each step : x <- ALPHA*x + a

struct decay_system {
  using state_type       = double;
  using command_type     = double;
  using observation_type = double
  using report_type      = gdyn::no_report;

  state_type x = 0;

  void operator=(state_type x_init) {x = x_init;}
  observation_type operator*() const {return x;}
  report_type operator()(command_type a) {
    x = ALPHA * x + a;
    return {};
  }
  operator bool() const {return true;}
};


// From this system, we what to set up a discrete system. Let us
// consider enumerable state as well as enumerable actions.

struct S_convertor {
  static constexpr std::size_t nb_bins {10};
  static double to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, 0., 100., nb_bins);}
  static std::size_t from(double value) {
    if(value < 0)     return 0;
    if(value >= 100.) return nb_bins - 1;
    return rl2::enumerable::utils::digitize::to_index(value, 0., 100., nb_bins);}
};
using S = rl2::enumerable::set<double, S_convertor::nb_bins, S_convertor>;


struct A_convertor {
  static double to(std::size_t index) {
    if(index == 0) return 0.;
    if(index == 1) return 1.;	 
    return 10;	     
  }
  static std::size_t from(continuous::interval value) {
    if(value > 1.) return 2;
    if(value > 0.) return 1;
    return 0;
  }
};

using A = rl2::enumerable::set<double, 3, A_convertor>;

// We can then define a discrete system from the continuous decay_system type.
using discrete_decay_system = rl2::enumerable::system<S, S, A, decay_system>;

