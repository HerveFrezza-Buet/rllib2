#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <algorithm>
#include <cstddef>

#include <gdyn.hpp>
#include <rllib2.hpp>

#include "discrete-rocket-problem.hpp"
#include "my_rocket_config.hpp"

#define NB_PASSES 1000

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Let us build up the encapsulations of our rocket.
  auto params = make_params();
  auto rocket = gdyn::problem::rocket::system(params);
  auto relative_rocket = gdyn::problem::rocket::relative::system(rocket, [target = params.ceiling_height/2](){return target;});
  auto exposed_rocket  = types::exposed_system(relative_rocket);
  // auto discrete_rocket = types::discrete_system(exposed_rocket); This one is useless here.

  // We apply Q-learning to get the best controller, using a simple
  // tabular Q function.
  std::array<double, types::SA::size()> values;
  auto Q = rl2::enumerable::make_two_args_tabular<types::S, types::A>(values.begin());

  // The idea here is to explore discrete initial SxA situations
  // systematically. Just for fun, we explore them several time with a
  // random permutation.
  std::array<std::size_t, types::SA::size()> permutation;
  std::size_t idx = 0;
  for(auto& i : permutation) i = idx++;

  std::cout << std::endl << std::endl
	    << "Computing Q-learning passes:" << std::endl;
  for(std::size_t pass = 0; pass < NB_PASSES; ++pass) {
    std::cout << std::setw(5) << pass+1 << '/' << NB_PASSES << "\r     " << std::flush;
    std::shuffle(permutation.begin(), permutation.end(), gen);
    for(auto [init_state, command]
	  : permutation
	  | std::views::transform([](auto sa_idx) {types::SA::iterator it {sa_idx}; return *it;})) {
      relative_rocket = init_state;
      auto reward = relative_rocket(command);
      auto next_state = *relative_rocket;
      // Nota : we do not need discrete_rocket since making transition
      // will do the cast into dicrete states and actions.
      rl2::sarsa<types::S, types::A> transition {init_state, command, reward, next_state};
      
    }
  }
  std::cout << "Done.                   " << std::endl;

  return 0;
}

  
