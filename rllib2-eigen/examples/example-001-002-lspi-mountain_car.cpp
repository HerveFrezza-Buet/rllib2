// uniform sampling of the problem
// example-001-002-lspi-mountain_car 1000 40 0.0 20.0

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
#include "mountain_car-defs.hpp"

// This fills a transition dataset.
// Draw 'nb_samples' starting states
// and apply a random policy for at most 'max_episode_length' steps
template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill_uniform(RANDOM_GENERATOR& gen, mountain_car& simulator, const POLICY& policy,
                   OutputIt out,
                   unsigned int nb_samples, unsigned int max_episode_length) {

  // parameters, defines starting point range
  auto params_uniform = gdyn::problem::mountain_car::parameters{};
  params_uniform.start_position_min = params_uniform.min_position;
  params_uniform.start_position_max = params_uniform.max_position;
  params_uniform.start_velocity_min = -params_uniform.max_speed;
  params_uniform.start_velocity_max =  params_uniform.max_speed;

  for (unsigned int episode=0; episode < nb_samples; ++episode) {
    simulator = gdyn::problem::mountain_car::random_state(gen, params_uniform);
    std::ranges::copy(gdyn::views::pulse(policy)
                      | gdyn::views::orbit(simulator)
                      | rl2::views::sarsa
                      | std::views::take(max_episode_length),
                      out);
  }
}
// This fills a transition dataset.
// starting points are taken from the problem settings
template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill(RANDOM_GENERATOR& gen, mountain_car& simulator, const POLICY& policy,
          OutputIt out,
          unsigned int nb_samples, unsigned int max_episode_length) {
  unsigned int to_be_filled = nb_samples;
  while(to_be_filled > 0) {
    simulator = gdyn::problem::mountain_car::random_state(gen, gdyn::problem::mountain_car::parameters());
    std::ranges::copy(gdyn::views::pulse(policy)
                      | gdyn::views::orbit(simulator)
                      | rl2::views::sarsa
                      | std::views::take(to_be_filled)
                      | std::views::take(max_episode_length)
                      | std::views::filter([&to_be_filled](const auto&){--to_be_filled; return true;}),
                      out);
  }
}


#define NB_TRANSITIONS        500
#define MAX_SAMPLE_LENGTH     10
#define MAX_EPISODE_LENGTH    300
#define NB_LSPI_ITERATIONS    1

#define NB_TEST_EPISODES        50
#define MAX_TEST_EPISODE_LENGTH  MAX_EPISODE_LENGTH

unsigned int nb_transitions = NB_TRANSITIONS;
unsigned int nb_lspi_iterations = NB_LSPI_ITERATIONS;

template<typename RANDOM_DEVICE, typename Q>
void test_policy(RANDOM_DEVICE& gen, const Q& q);

template<typename Q>
void run_policy(S start, const Q& q);

// *****************************************************************************
//                                                                          MAIN
// *****************************************************************************
int main(int argc, char *argv[])
{

  // with arguments : nb_trans, nb_iter, alpha, tau_epsilon uniform?
  if (argc < 5) {
    std::cout << "usage: " << argv[0] << " nb_trans nb_iter alpha tau_epsilon ['on_traj']" << std::endl;
    return 1;
  }

  nb_transitions = std::stoi( argv[1] );
  nb_lspi_iterations = std::stoi( argv[2] );
  double alpha = std::stod( argv[3] );
  double epsilon = 1.0;
  double tau_epsilon = std::stod( argv[4] );
  double epsilon_threshold    = 0.0001;
  bool uniform_sampling = true;
  if (argc == 6) {
    uniform_sampling = false;
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  // This will store transitions for LSTDQ computation.
  std::vector<rl2::sarsa<S, A>> transitions;

  auto sim = gdyn::problem::mountain_car::make();
  mountain_car simulator {sim}; // simulator is sim, but handling discrete actions.

  // First, we fill the dataset with a random policy
  std::cout << "Filling the dataset... " << std::flush;
  if (uniform_sampling) {
    // applied on 'nb_transition' uniformly sampled starting points
    fill_uniform(gen, simulator,
                 rl2::discrete::uniform_sampler<A>(gen),
                 std::back_inserter(transitions),
                 nb_transitions, MAX_SAMPLE_LENGTH);
  }
  else {
    // applied on 'nb_transition' starting points chosen according to problem
    fill(gen, simulator,
         rl2::discrete::uniform_sampler<A>(gen),
         std::back_inserter(transitions),
         nb_transitions, MAX_EPISODE_LENGTH);
  }
  std::cout << " got " << transitions.size() << " samples, ("
            << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
            << " are terminal transitions)." << std::endl;

  // For LSPI, we need a parametrized Q function, and related policies.
  Q q      {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};
  Q next_q {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};

  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " features"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;


  // Let us initialize Q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double gamma = .95;
  double error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
						      rl2::discrete_a::random_policy<S, A>(gen),
						      gamma,
						      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  test_policy(gen, next_q);
  // copy weights from next_q  to q can be achieved with a swap
  std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.
  // test_policies
  test_policy(gen, q);

  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 0; i < nb_lspi_iterations; ++i) {

    if (uniform_sampling) {
      // no need to re-sample the transitions set, as they can be re-used.
    }
    else {
      // TODO améliorer le remplissage de transitions
      // sample using random policy AND epsilon_greedy
      transitions.clear();
      fill(gen, simulator,
           rl2::discrete::uniform_sampler<A>(gen),
           std::back_inserter(transitions),
           nb_transitions/2, MAX_EPISODE_LENGTH);
      fill(gen, simulator,
           [&simulator, &epsilon_greedy_on_q](){return epsilon_greedy_on_q(*simulator);},
           std::back_inserter(transitions),
           nb_transitions/2, MAX_EPISODE_LENGTH);

    }

    auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
							     // epsilon_greedy_on_q,
							    greedy_on_q,
							    gamma,
							    transitions.begin(), transitions.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    // It is also possible to use a progressive change in params
    // q.params <- alpha x q.params + (1 - alpha) x next_q.params
    // q.params are nupplet
    // TODO use range to do the following ?
    auto q_params_it = q.params->begin();
    auto next_q_params_it = next_q.params->begin();
    for ( ; q_params_it != q.params->end(); ) {
      auto new_w = alpha * (*q_params_it) + (1.0 - alpha) * (*next_q_params_it);
      *q_params_it = new_w;
      ++q_params_it;
      ++next_q_params_it;
    }


    test_policy(gen, q);

    epsilon *= 1.0 / tau_epsilon;
    epsilon = std::max( epsilon, epsilon_threshold );
  }

  // test_policy(gen, next_q);

  return 0;
}

template<typename RANDOM_DEVICE, typename Q>
void test_policy(RANDOM_DEVICE& gen, const Q& q) {
  auto sim = gdyn::problem::mountain_car::make();
  mountain_car simulator {sim}; // simulator is sim, but handling discrete actions.

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(unsigned int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    unsigned int length = 0;
    simulator = gdyn::problem::mountain_car::random_state(gen, gdyn::problem::mountain_car::parameters());
    for([[maybe_unused]] const auto& unused
			   : gdyn::views::controller(simulator, rl2::discrete::greedy_ify(q))
			   | gdyn::views::orbit(simulator)
			   | std::views::take(MAX_TEST_EPISODE_LENGTH))
      ++length;
    if(length < (MAX_TEST_EPISODE_LENGTH))
      nb_success ++;
    sum_length += length;
  }
  std::cout << std::endl
	    << "Success on " << int(100*nb_success/NB_TEST_EPISODES+.5) << "% of the episodes," << std::endl 
	    << "           (average episode length is " << sum_length/NB_TEST_EPISODES << ")." << std::endl;

}

template<typename Q>
void run_policy(S start, const Q& q) {
  auto sim = gdyn::problem::mountain_car::make();
  mountain_car simulator {sim}; // simulator is sim, but handling discrete actions.
  simulator = start;
  std::cout << "RUN:" << std::endl;
  for(auto [s, a, r, next_s, next_a]
        : gdyn::views::controller(simulator, rl2::discrete::greedy_ify(q))
        | gdyn::views::orbit(simulator)
        | rl2::views::sarsa
        | std::views::take(MAX_TEST_EPISODE_LENGTH)) 
    std::cout << "  " << s << " --> " << static_cast<A::base_type>(a) << std::endl;
}

