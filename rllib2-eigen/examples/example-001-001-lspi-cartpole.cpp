#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <iterator>
#include <tuple>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

#include <gdyn.hpp>
#include <rllib2.hpp>
#include <rllib2-eigen.hpp>

#include "trace-utils.hpp"

// Read the type definitions in this file
#include "cartpole-utils.hpp"

using c_system = utils::cartpole::system;
using c_S = c_system::state_type;
using c_A = c_system::command_type;
using c_s_feature = utils::cartpole::s_feature;
using c_params = utils::cartpole::params;
using c_Q = utils::cartpole::Q;

// Draw 'nb_samples' starting states
// and apply a random policy for at most 'max_episode_length' steps
template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill_uniform(RANDOM_GENERATOR& gen, c_system& simulator, const POLICY& policy,
                   OutputIt out,
                   unsigned int nb_samples, unsigned int max_episode_length) {

  // parameters, defines starting point range as the maximal range
  auto params = gdyn::problem::cartpole::parameters{};

  for (unsigned int episode=0; episode < nb_samples; ++episode) {
    // uniform density on state space
    gdyn::problem::cartpole::state s;
    s.x = (std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0) * 1.5 * params.x_threshold;
    s.x_dot = (std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0) * 10.0;
    s.theta = (std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0) * 1.5 * params.theta_threshold_rad;
    s.theta_dot = (std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0) * 5.0;
    simulator = s;

    std::ranges::copy(gdyn::views::pulse(policy)
                      | gdyn::views::orbit(simulator)
                      | rl2::views::sarsa
                      | std::views::take(max_episode_length),
                      out);
  }
}


#define NB_TRANSITIONS        500
#define MAX_SAMPLE_LENGTH     10
#define MAX_EPISODE_LENGTH    200
#define NB_LSPI_ITERATIONS    10

#define NB_TEST_EPISODES        20
#define MAX_TEST_EPISODE_LENGTH  MAX_EPISODE_LENGTH

#define GAMMA                  0.95

// Declaration of some helpers function to trace activity ***********************
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                   const POLICY& policy,
                   std::ostream& file_csv,
                   utils::trace::csv<>& g_trace, int ite_nb);

void store_transitions(const std::vector<rl2::sarsa<c_S, c_A>>& transitions,
                     std::ostream& file_csv );

// *****************************************************************************
//                                                                          MAIN
// *****************************************************************************
int main(int argc, char *argv[]) {

  std::random_device rd;
  std::mt19937 gen(rd());

  // This will store transitions for LSTDQ computation.
  std::vector<rl2::sarsa<c_S, c_A>> transitions;
  
  auto sim = gdyn::problem::cartpole::make();
  c_system simulator {sim}; // simulator is sim, but handling discrete actions.

  // For LSPI, we need a parametrized Q function, and related policies.
  c_Q q      {std::make_shared<c_s_feature>(utils::cartpole::make_state_feature()), std::make_shared<c_params>()};
  c_Q next_q {std::make_shared<c_s_feature>(utils::cartpole::make_state_feature()), std::make_shared<c_params>()};

  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << c_A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " feature"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;

  // The transition buffer is sampled once from the entire space state, uniformely.
  // At each of the NB_TRANSITIONS starting point,
  // we store MAX_SAMPLE_LENGTH transitions using a random policy,
  // this transition buffer is used in every iteration of LSPI.
  std::cout << "Filling the dataset with " << NB_TRANSITIONS << " sampled starting states..." << std::flush;
  fill_uniform(gen, simulator,
               rl2::discrete::uniform_sampler<c_A>(gen),
               std::back_inserter(transitions),
               NB_TRANSITIONS, MAX_SAMPLE_LENGTH);
  {
    std::ofstream local_file {"transition_cartpole_lspi_0.csv"};
    store_transitions( transitions, local_file );
  }

  std::cout << " got " << transitions.size() << " samples, ("
            << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
            << " are terminal transitions)." << std::endl;

  std::ofstream global_file {"global_cartpole.csv"};
  utils::trace::csv<> global_trace {global_file};

  std::array<std::string, 3> header {"## it", "ep", "len"};
  global_trace += header;

  // Let us initialize q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double error = rl2::eigen::critic::discrete_a::lstd<true>(q,
                      rl2::discrete_a::random_policy<c_S, c_A>(gen),
                      GAMMA,
                      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  // test this initial policy
  {
    std::ofstream local_file {"test_cartpole_lspi_0.csv"};
    test_policy(gen, greedy_on_q,
                local_file, global_trace, 0);
    std::ofstream qval_file {"qval_cartpole_lspi_0.csv"};
  }

  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 1; i <= NB_LSPI_ITERATIONS; ++i) {

    auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
                      greedy_on_q,
                      GAMMA,
                      transitions.begin(), transitions.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    // We can simply update 'q' using the weights of 'next_q' using
    std::swap(q, next_q);

    {
      std::ofstream local_file {std::string("test_cartpole_lspi_")+std::to_string(i)+".csv"};
      test_policy(gen, greedy_on_q,
                  local_file, global_trace, i);
      std::ofstream qval_file {std::string("qval_cartpole_lspi_")+std::to_string(i)+".csv"};
    }

  }

  return 0;
}

// *****************************************************************************
// Helper functions to trace activity ******************************************

/** Run episods and trace results in file_csv
 *
 * traces globaly : iteration_nb episode_nb length
 * traces localy : episode_nb t s.x s.x_dot s.thet s.theta_dot a
 *                              next_s.x next_s.x_dot next_s.theta next_s.theta_dot r
 */
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                 const POLICY& policy,
                 std::ostream& file_csv,
                 utils::trace::csv<>& g_trace, int ite_nb)
{
  auto sim = gdyn::problem::cartpole::make();
  c_system simulator {sim}; // simulator is sim, but handling discrete actions.

  // trace with header
  utils::trace::csv<> test {file_csv};
  std::array<std::string, 12> header {"## ep", "t", "s.x", "s.x_dot", "s.theta", "s.theta_dot",
                                     "a",
                                     "next_s.x", "next_s.x_dot", "next_s.theta", "next_s.theta_dot",
                                     "r"};
  test += header;

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    int length = 0;
    simulator = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
    for(auto [s, a, r, next_s, next_a]
          : gdyn::views::controller(simulator, policy)
          | gdyn::views::orbit(simulator)
          | rl2::views::sarsa
          | std::views::take(MAX_TEST_EPISODE_LENGTH)) {
      std::array<double, 12> line  { static_cast<double>(episode),
                                    static_cast<double>(length),
                                    s.x, s.x_dot,
                                    s.theta, s.theta_dot,
                                    static_cast<double>(a),
                                    next_s.x, next_s.x_dot,
                                    next_s.theta, next_s.theta_dot,
                                    r };
      test += line;

      ++length;
    }

    std::array<int, 3> line { ite_nb, episode, length };
    g_trace += line;

    if(length == (MAX_TEST_EPISODE_LENGTH))
      nb_success ++;
    sum_length += length;
  }
  std::cout << std::endl
            << "Success on " << int(100*nb_success/NB_TEST_EPISODES+.5) << "% of the episodes," << std::endl
            << "           (average episode length is " << sum_length/NB_TEST_EPISODES << ")." << std::endl;

}
/** Traces all the transistions
 *
 * traces localy : episode_nb t s.x s.x_dot s.thet s.theta_dot a
 *                              next_s.x next_s.x_dot next_s.theta next_s.theta_dot r
 */
void store_transitions(const std::vector<rl2::sarsa<c_S, c_A>>& transitions,
                       std::ostream& file_csv )
{
  // trace with header
  utils::trace::csv<> trace {file_csv};
  std::array<std::string, 9> header {"## s.x", "s.x_dot", "s.theta", "s.theta_dot",
                                     "a",
                                     "next_s.x", "next_s.x_dot", "next_s.theta", "next_s.theta_dot" };
  trace += header;

  for (auto t : transitions ) {
      std::array<double, 9> line {
        t.s.x, t.s.x_dot,
        t.s.theta, t.s.theta_dot,
        static_cast<double>(t.a),
        t.ss.x, t.ss.x_dot,
        t.ss.theta, t.ss.theta_dot };

      trace += line;
  }
}
