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

// Log results in traces
#include <utils/trace.hpp>

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


#define NB_TRANSITIONS        500
#define MAX_EPISODE_LENGTH    100
#define NB_LSPI_ITERATIONS    2000


template<typename RANDOM_DEVICE, typename Q>
void test_policy(RANDOM_DEVICE& gen, const Q& q);

template<typename Q>
void run_policy(S start, const Q& q);

template<typename RANDOM_DEVICE, typename POLICY>
void test_policy_with_trace(RANDOM_DEVICE& gen,
                            const POLICY& policy,
                            const std::string& trace_name,
                            utils::Trace& g_trace, int ite_nb);

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
  // Q q      {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};
  // Q next_q {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};

  raw_Q q      {std::make_shared<raw_s_feature>(), std::make_shared<raw_params>()};
  raw_Q next_q {std::make_shared<raw_s_feature>(), std::make_shared<raw_params>()};
  // TODO j'ai eu du mal à retrouver la def rl2::linear::discrete_a::q
  // (pour q.s_feature)
  q.s_feature->data = std::make_shared<std::array<double, raw_s_feature::dim>>();
  next_q.s_feature->data = std::make_shared<std::array<double, raw_s_feature::dim>>();


  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  double epsilon             = 1.0;
  double tau_epsilon         = 100.0;
  double epsilon_threshold    = 0.0001;
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " features"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;


  // Trace to log test statistics
  utils::Trace global_trace;
  std::stringstream header;
  header << "it\tep\tlen";
  global_trace.add_header( header.str() );

  // test random policy
  test_policy_with_trace(gen, rl2::discrete_a::random_policy<S, A>(gen),
                         "test_rnd", global_trace, -2);

  // Let us initialize Q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double gamma = .9;
  double error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
						      rl2::discrete_a::random_policy<S, A>(gen),
						      gamma,
						      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  // test_policy(gen, next_q);
  test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
                         "test_init", global_trace, -1);
  
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

    test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
                           "test_lspi_"+std::to_string(i), global_trace, i);

    epsilon *= 1.0 / tau_epsilon;
    epsilon = std::max( epsilon, epsilon_threshold );
  }

  std::cout << "Saving global_trace in lerning.csv" << std::endl;
  std::ofstream outfile( "learning.csv" );
  utils::Trace::write( outfile, global_trace );
  outfile.close();

  // test_policy(gen, next_q);
  run_policy(S(), next_q);
  
  return 0;
}

#define NB_TEST_EPISODES        10
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
    if(length == (MAX_TEST_EPISODE_LENGTH))
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

/** Run episods and store results in file
 *
 * episode_nb t s.x s.x_dot s.theta s.theta_dot a next_s.x ... next_s.theta_dot r
 *
 * Returns: nb_success (int)
 */
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy_with_trace(RANDOM_DEVICE& gen,
                            const POLICY& policy,
                            const std::string& trace_name,
                            utils::Trace& g_trace, int ite_nb)
{
  auto sim = gdyn::problem::cartpole::make();
  cartpole simulator {sim}; // simulator is sim, but handling discrete actions.

  // trace with header
  std::stringstream header;
  header << "ep\tt\ts.x\ts.x_dot\ts.theta\ts.theta_dot\ta";
  header << "\tnext_s.x\tnext_s.x_dot\tnext_s.theta\tnext_s.theta_dot\tr";
  utils::Trace local_trace;
  local_trace.add_header( header.str() );

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(unsigned int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    unsigned int length = 0;
    simulator = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
    for(auto [s, a, r, next_s, next_a]
          : gdyn::views::controller(simulator, policy)
          | gdyn::views::orbit(simulator)
          | rl2::views::sarsa
          | std::views::take(MAX_TEST_EPISODE_LENGTH)) {

      local_trace.push_to_state( static_cast<double>(episode) );
      local_trace.push_to_state( static_cast<double>(length) );
      local_trace.push_to_state( s.x );
      local_trace.push_to_state( s.x_dot );
      local_trace.push_to_state( s.theta );
      local_trace.push_to_state( s.theta_dot );
      local_trace.push_to_state( static_cast<double>(a) );
      local_trace.push_to_state( next_s.x );
      local_trace.push_to_state( next_s.x_dot );
      local_trace.push_to_state( next_s.theta );
      local_trace.push_to_state( next_s.theta_dot );
      local_trace.push_to_state( r );
      local_trace.store_state();

      ++length;

    }
    g_trace.push_to_state( static_cast<double>(ite_nb) );
    g_trace.push_to_state( static_cast<double>(episode) );
    g_trace.push_to_state( static_cast<double>(length) );
    g_trace.store_state();

    if(length == (MAX_TEST_EPISODE_LENGTH))
      nb_success ++;
    sum_length += length;
  }
  std::cout << std::endl
	    << "Success on " << int(100*nb_success/NB_TEST_EPISODES+.5) << "% of the episodes," << std::endl
	    << "           (average episode length is " << sum_length/NB_TEST_EPISODES << ")." << std::endl;

  std::cout << "Saving trace in " << trace_name << ".csv" << std::endl;
  std::ofstream outfile( trace_name+".csv" );
  utils::Trace::write( outfile, local_trace );
  outfile.close();
}
