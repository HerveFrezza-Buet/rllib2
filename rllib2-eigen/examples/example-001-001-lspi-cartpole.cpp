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

// This fills a transition dataset.
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


#define NB_TRANSITIONS        2
#define MAX_EPISODE_LENGTH   20
#define NB_LSPI_ITERATIONS    0

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
       NB_TRANSITIONS, MAX_EPISODE_LENGTH);
  std::cout << " got " << transitions.size() << " samples, (" 
	    << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
	    << " are terminal transitions)." << std::endl;

  // For LSPI, we need a parametrized Q function, and related policies.
  Q q      {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};
  Q next_q {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};
  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  double epsilon             = .1;
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "__ Q est un combinaison linéaire de RBF" << std::endl;
  // TODO pourquoi ce 2 il est pas quelque part dans la def du probleme
  std::cout << "   with continuous state space and " << 2 << "discrete actions" << std::endl;
  // TODO j'en ai chié, pk *(q.s_feature)(it->s) marche pas ??
  std::cout << "   " << q.s_feature->dim << " features"<< std::endl;
  std::cout << "   " << q.params->dim << " parameters" << std::endl;

  // Let us initialize Q from the random policy.
  rl2::eigen::critic::discrete_a::lstd(next_q,
				       rl2::discrete_a::random_policy<S, A>(gen),
				       transitions.begin(), transitions.end());

  // LSPI iteration
  for(unsigned int i = 0; i < NB_LSPI_ITERATIONS; ++i) {
    std::swap(q, next_q);
    rl2::eigen::critic::discrete_a::lstd(next_q,
					 epsilon_greedy_on_q,
					 transitions.begin(), transitions.end());
  }
  

  return 0;
}
