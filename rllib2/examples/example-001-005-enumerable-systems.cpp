#include <cmath>
#include <cstddef>
#include <numbers>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <ranges>
#include <string>
	

#include <rllib2.hpp>

#define ALPHA .5

// Let us consider a dynamical system made of 
// single scalar 'x' as a state. The command 'a' is a scalar as well
// (both are positive). At each step : x <- ALPHA*x + a

struct decay_system {
  using state_type       = double;
  using command_type     = double;
  using observation_type = double;
  using report_type      = double; // RL system must provide a reward.

  state_type x = 0;

  void operator=(state_type x_init) {x = x_init;}
  observation_type operator*() const {return x;}
  report_type operator()(command_type a) {
    x = ALPHA * x + a;
    return 0.; // We always return a null reward.
  }
  operator bool() const {return true;}
};


// From this system, we what to set up a discrete system. Let us
// consider enumerable state as well as enumerable actions.

#define MAX_X 10.
struct S_convertor {
  static constexpr std::size_t nb_bins {10};
  static double to(std::size_t index)   {return rl2::enumerable::utils::digitize::to_value(index, 0., MAX_X, nb_bins);}
  static std::size_t from(double value) {
    if(value < 0)      return 0;
    if(value >= MAX_X) return nb_bins - 1;
    return rl2::enumerable::utils::digitize::to_index(value, 0., MAX_X, nb_bins);}
};
using S = rl2::enumerable::set<double, S_convertor::nb_bins, S_convertor>;


struct A_convertor {
  static double to(std::size_t index) {
    if(index == 0) return 0.;
    if(index == 1) return 1.;	 
    return 10;	     
  }
  static std::size_t from(double value) {
    if(value > 1.) return 2;
    if(value > 0.) return 1;
    return 0;
  }
};

using A = rl2::enumerable::set<double, 3, A_convertor>;
// A is serializable already (into a integer, which is the index). Let
// us provide a more informative serialization.
std::string to_string(A a) {
  switch(static_cast<std::size_t>(a)) {
  case 0: return "none";
  case 1: return "small-jump";
  default: return "big-jump";
  }
}


// We can then define a discrete system from the continuous decay_system type.
using discrete_decay_system = rl2::enumerable::system<S, S, A, decay_system>;


#define NB_ACTIONS 10

int main(int argc, char* argv[]) {
  // First let us set up a decay system, and wrapp around it a discrete one.

  decay_system           system;
  discrete_decay_system dsystem {system};
  
  // Let us define a sequence of discrete actions.
  auto a_it = A::begin();
  A none       {a_it++};
  A small_jump {a_it++};
  A big_jump   {a_it++};
  std::array<A, NB_ACTIONS+1> actions {
    small_jump,
    none,
    none,
    none,
    big_jump,
    none,
    none,
    small_jump,
    none,
    none,
    big_jump,
    };

  // Let us run the discrete system and register transitions (with
  // discrete states and actions).
  system = .5; // continuous state initialization
  for(auto [s, a, r, ss, aa]
	: actions
	| gdyn::views::orbit(dsystem) 
	| rl2::views::sarsa
	| std::views::take(NB_ACTIONS))
    std::cout << "s = " << std::setw(3) << s << ", "
	      << "a = " << std::setw(10) << to_string(a) << ", "
	      << "r = " << r << ", "
	      << "s' = " << ss  << ", "
	      << "a' = " << to_string(*aa) << std::endl;

  // In such a wrapped system, we may be interested in what happens to
  // the base system, i.e. collectig transitions corresponding to it
  // (with continuous states and actions), while using the same
  // policy (i.e the same actions table here).

  
  return 0;
}




