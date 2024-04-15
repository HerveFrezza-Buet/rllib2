#include <iostream>
#include <iomanip>
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


#define NB_TRANSITIONS        200
#define MAX_EPISODE_LENGTH    100
#define NB_LSPI_ITERATIONS    100


template<typename RANDOM_DEVICE, typename Q>
void test_policy(RANDOM_DEVICE& gen, const Q& q);

template<typename Q>
void run_policy(S start, const Q& q);

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

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " features"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;

  // Let us initialize Q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double gamma = .9;
  double error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
						      rl2::discrete_a::random_policy<S, A>(gen),
						      gamma,
						      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  test_policy(gen, next_q);
  
  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 0; i < NB_LSPI_ITERATIONS; ++i) {
    std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.

    // We re-sample the transition set, following the current q-values
    // with an epsilon-greedy policy.
    transitions.clear();
    fill(gen, simulator,
	 [&simulator, &epsilon_greedy_on_q](){return epsilon_greedy_on_q(*simulator);},
	 std::back_inserter(transitions),
	 NB_TRANSITIONS, MAX_EPISODE_LENGTH);
    
    auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
							    epsilon_greedy_on_q,
							    gamma,
							    transitions.begin(), transitions.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;
  }

  test_policy(gen, next_q);

  run_policy(S(), next_q);
  
  return 0;
}

#define NB_TEST_EPISODES        1000
#define MAX_TEST_EPISODE_LENGTH  MAX_EPISODE_LENGTH
template<typename RANDOM_DEVICE, typename Q>
void test_policy(RANDOM_DEVICE& gen, const Q& q) {
  auto sim = gdyn::problem::cartpole::make();
  cartpole simulator {sim}; // simulator is sim, but handling discrete actions.

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(unsigned int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    unsigned int length = 0;
    simulator = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
    for([[maybe_unused]] const auto& unused
			   : gdyn::views::controller(simulator, rl2::discrete::greedy_ify(q))
			   | gdyn::views::orbit(simulator)
			   | std::views::take(MAX_TEST_EPISODE_LENGTH))
      ++length;
    if(length == MAX_TEST_EPISODE_LENGTH)
      nb_success ++;
    sum_length += length;
  }
  std::cout << std::endl
	    << "Success on " << int(100*nb_success/NB_TEST_EPISODES+.5) << "% of the episodes," << std::endl 
	    << "           (average episode length is " << sum_length/NB_TEST_EPISODES << ")." << std::endl;

}

template<typename Q>
void run_policy(S start, const Q& q) {
  auto sim = gdyn::problem::cartpole::make();
  cartpole simulator {sim}; // simulator is sim, but handling discrete actions.
  simulator = start;
  std::cout << "RUN:" << std::endl;
  for(auto [s, a, r, next_s, next_a]
        : gdyn::views::controller(simulator, rl2::discrete::greedy_ify(q))
        | gdyn::views::orbit(simulator)
        | rl2::views::sarsa
        | std::views::take(MAX_TEST_EPISODE_LENGTH)) 
    std::cout << "  " << s << " --> " << static_cast<A::base_type>(a) << std::endl;
}
