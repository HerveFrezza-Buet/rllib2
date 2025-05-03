#include <iostream>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

#include "discrete-rocket-problem.hpp"
#include "my_rocket_config.hpp"

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Let us build up the encapsulations of our rocket.
  gdyn::problem::rocket::parameters params;
  auto rocket = gdyn::problem::rocket::system(params);
  auto relative_rocket = gdyn::problem::rocket::relative::system(rocket, [target = params.ceiling_height/2](){return target;});
  auto exposed_rocket  = types::exposed_system(relative_rocket);
  auto discrete_rocket = types::discrete_system(exposed_rocket);

  // We apply Q-learning to get the best controller, using a simple
  // tabular Q function.
  std::array<double, types::SA::size()> values;
  auto Q = rl2::enumerable::make_two_args_tabular<types::S, types::A>(values.begin());

  // Let us define learning policies.
  
  

  return 0;
}

  
