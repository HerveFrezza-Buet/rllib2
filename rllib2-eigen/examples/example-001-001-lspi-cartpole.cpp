#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <tuple>
#include <algorithm>

#include <Eigen/Dense>

#include <rllib2.hpp>
#include <rllib2-eigen.hpp>

#include <gdyn.hpp>

// Read the type definitions in this file
#include "cartpole-defs.hpp"

// Eigen::Vector<double, 4>;



template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill(RANDOM_GENERATOR& gen, cartpole& simulator, const POLICY& policy,
	  OutputIt out,
	  unsigned int nb_samples, unsigned int max_episode_length) {
  unsigned int to_be_filled = nb_samples;      
  while(to_be_filled > 0) {
    simulator = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
    std::ranges::copy(gdyn::views::pulse(policy)
     		      | gdyn::views::orbit(simulator)  
		      | rl2::views::sarsa
		      | std::views::take(to_be_filled)
		      | std::views::take(max_episode_length)
		      | std::views::filter([&to_be_filled](const auto&){--to_be_filled; return true;}),
		      out);
  }
}


#define NB_TRANSITIONS     1000
#define MAX_EPOSODE_LENGTH   20

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // This will store transitions for LSTDQ computation.
  std::vector<rl2::sarsa<S, A>> transitions;
  
  auto sim = gdyn::problem::cartpole::make();
  cartpole simulator {sim}; // simulator is sim, but handling discrete actions.

  // First, we fill the dataset with a random policy.
  std::cout << "Filling the dataset... " << std::flush;
  fill(gen, simulator,
       rl2::discrete::uniform_sampler<A>(gen),
       std::back_inserter(transitions),
       NB_TRANSITIONS, MAX_EPOSODE_LENGTH);
  std::cout << " got " << transitions.size() << " samples, (" 
	    << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
	    << " are terminal transitions)." << std::endl;

  // For lspi, we need a parametrized Q function, and related policies.
  auto phi = make_features;

  

  return 0;
}
