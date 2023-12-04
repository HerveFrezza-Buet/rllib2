/**
  Explore la distribution de x_dot et theta_dot dans le cartpole
  */

#include <iostream>
#include <limits>
#include <random>
#include <rllib2.hpp>

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // continuous cartpole environment
  using cartpole_type = gdyn::problem::cartpole::system;
  auto sys = gdyn::problem::cartpole::make();

  unsigned int epoch_nb = 5;
  unsigned int epoch_max_length = 200;
  unsigned int samples_nb = 0;

  double x_dot_min = std::numeric_limits<double>::max();
  double x_dot_max = std::numeric_limits<double>::lowest();

  // random start, random controler
  auto rnd_policy = [&gen](const cartpole_type::observation_type) {
    return gdyn::problem::cartpole::random_command(gen);
  };

  // system initialization
  sys = cartpole_type::state_type(0,0,0,0);
  samples_nb = 0;
  unsigned int epoch_length = 0;
  for(auto [o, a, r]
        // TODO dire/expliquer que le rl2::ranges ne marche que sur les MDP
        : gdyn::ranges::controller(sys, rnd_policy)
        | gdyn::views::orbit(sys)
        | std::views::take(epoch_max_length)) {
    samples_nb += 1;
    epoch_length += 1;
    x_dot_min = std::min(o.x_dot, x_dot_min);
    x_dot_max = std::max(o.x_dot, x_dot_max);
  }
  std::cout << epoch_length << ", " << std::endl;

  std::cout << "After " << samples_nb << " samples we have:\n"
            << "  x_dot in [" << x_dot_min << ", " << x_dot_max << "]"
            << std::endl;

  return 0;
}
