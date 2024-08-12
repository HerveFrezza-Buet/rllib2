// Using this example
//
// * generate CVS file for offline and online LSPI
// example-001-002-lspi-mountain-car
//
// * make nice plot using R => fig_offline.png and fig_online.png and some more...
// Rscript $PATH_TO_SCRIPTS/view_results.R

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


// Read the type definitions in this file
#include "mountain-car-utils.hpp"
#include "trace-utils.hpp"

using mc_system = utils::mountain_car::system;
using mc_S = mc_system::state_type;
using mc_A = mc_system::command_type;
using mc_s_feature = utils::mountain_car::s_feature;
using mc_params = utils::mountain_car::params;
using mc_Q = utils::mountain_car::Q;

// This fills a transition dataset.
// Draw 'nb_samples' starting states
// and apply a random policy for at most 'max_episode_length' steps
template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill_uniform(RANDOM_GENERATOR& gen, mc_system& simulator, const POLICY& policy,
                   OutputIt out,
                   unsigned int nb_samples, unsigned int max_episode_length) {

  // parameters, defines starting point range as the maximal range
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
void fill(RANDOM_GENERATOR& gen, mc_system& simulator, const POLICY& policy,
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
#define NB_LSPI_ITERATIONS    10

#define NB_TEST_EPISODES        50
#define MAX_TEST_EPISODE_LENGTH  MAX_EPISODE_LENGTH

#define Q_SAMPLE_DENSITY       100

#define GAMMA                  0.95

#define EPSILON_START          1.0
#define EPSILON_TAU            0.1
#define EPSILON_THRESHOLD      0.0001

#define ALPHA                  0.05

// Declaration of some helpers function to trace activity ***********************
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                   const POLICY& policy,
                   std::ostream& file_csv,
                   utils::trace::csv<>& g_trace, int ite_nb);

template<typename Q>
void sample_V_and_pi(const Q& q,
                     std::ostream& file_csv,
                     int nb_bins);

void store_transitions(const std::vector<rl2::sarsa<mc_S, mc_A>>& transitions,
                     std::ostream& file_csv );

// *****************************************************************************
//                                                                          MAIN
// *****************************************************************************
int main(int argc, char *argv[])
{

  double epsilon = EPSILON_START;

  std::random_device rd;
  std::mt19937 gen(rd());
  
  // This will store transitions for LSTDQ computation.
  std::vector<rl2::sarsa<mc_S, mc_A>> transitions;
  
  auto sim = gdyn::problem::mountain_car::make();
  mc_system simulator {sim}; // simulator is sim, but handling discrete actions.

  // For LSPI, we need a parametrized Q function, and related policies.
  mc_Q q      {std::make_shared<mc_s_feature>(utils::mountain_car::make_state_feature()), std::make_shared<mc_params>()};
  mc_Q next_q {std::make_shared<mc_s_feature>(utils::mountain_car::make_state_feature()), std::make_shared<mc_params>()};

  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << mc_A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " feature"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;

  // ***************************************************************************
  // First, an easy scenario.
  {
  std::cout << "***** EASY SETTING : offline sampling **************************" << std::endl;
  // The transition buffer is sampled once from the entire space state, uniformely.
  // At each of the NB_TRANSITIONS starting point,
  // we store MAX_SAMPLE_LENGTH transitions using a random policy,
  // this transition buffer is used in every iteration of LSPI.
  std::cout << "Filling the dataset with " << NB_TRANSITIONS << " sampled starting states..." << std::flush;
  fill_uniform(gen, simulator,
               rl2::discrete::uniform_sampler<mc_A>(gen),
               std::back_inserter(transitions),
               NB_TRANSITIONS, MAX_SAMPLE_LENGTH);
  {
    std::ofstream local_file {"transition_offline_lspi_0.csv"};
    store_transitions( transitions, local_file );
  }

  std::cout << " got " << transitions.size() << " samples, ("
            << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
            << " are terminal transitions)." << std::endl;
  

  std::ofstream global_file {"global_offline.csv"};
  utils::trace::csv<> global_trace {global_file};

  std::array<std::string, 3> header {"## it", "ep", "len"};
  global_trace += header;

  // Let us initialize q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double error = rl2::eigen::critic::discrete_a::lstd<true>(q,
                      rl2::discrete_a::random_policy<mc_S, mc_A>(gen),
                      GAMMA,
                      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  // test this initial policy
  {
    std::ofstream local_file {"test_offline_lspi_0.csv"};
    test_policy(gen, greedy_on_q,
                local_file, global_trace, 0);
    std::ofstream qval_file {"qval_offline_lspi_0.csv"};
    sample_V_and_pi(q, qval_file, Q_SAMPLE_DENSITY);
  }

  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 1; i <= NB_LSPI_ITERATIONS; ++i) {

    auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
                      greedy_on_q,
                      GAMMA,
                      transitions.begin(), transitions.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    // We could simply update 'q' using the weights of 'next_q' using
    std::swap(q, next_q);

    {
      std::ofstream local_file {std::string("test_offline_lspi_")+std::to_string(i)+".csv"};
      test_policy(gen, greedy_on_q,
                  local_file, global_trace, i);
      std::ofstream qval_file {std::string("qval_offline_lspi_")+std::to_string(i)+".csv"};
      sample_V_and_pi(q, qval_file, Q_SAMPLE_DENSITY);
    }

  }
  } // end of easy setting

  // ***************************************************************************
  // Then a difficult scenario
  {
  std::cout << "***** HARD SETTING : online sampling ***************************" << std::endl;
  // The transition buffer is sampled using an 'exploration' policy.
  // At start, this is a random policy.
  // Then a epsilon-greedy policy on the current QValue estimation.
  // We repeat sampling until we get MAX_SAMPLE_LENGTH * NB_TRANSITION samples
  // this transition buffer is *updated* after every iteration of LSPI.
  std::cout << "Filling the dataset with " << MAX_SAMPLE_LENGTH*NB_TRANSITIONS << " sampled..." << std::flush;
  transitions.clear();
  fill(gen, simulator,
       rl2::discrete::uniform_sampler<mc_A>(gen),
       std::back_inserter(transitions),
       NB_TRANSITIONS * MAX_SAMPLE_LENGTH, MAX_EPISODE_LENGTH);
  {
    std::ofstream local_file {"transition_online_lspi_0.csv"};
    store_transitions( transitions, local_file );
  }

  std::cout << " got " << transitions.size() << " samples, ("
            << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
            << " are terminal transitions)." << std::endl;

  std::ofstream global_file {"global_online.csv"};
  utils::trace::csv<> global_trace {global_file};

  std::array<std::string, 3> header {"## it", "ep", "len"};
  global_trace += header;

  // Let us initialize q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double error = rl2::eigen::critic::discrete_a::lstd<true>(q,
                      rl2::discrete_a::random_policy<mc_S, mc_A>(gen),
                      GAMMA,
                      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  // test this initial policy
  {
    std::ofstream local_file {"test_online_lspi_0.csv"};
    test_policy(gen, greedy_on_q,
                local_file, global_trace, 0);
    std::ofstream qval_file {"qval_online_lspi_0.csv"};
    sample_V_and_pi(q, qval_file, Q_SAMPLE_DENSITY);
  }

  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 1; i <= NB_LSPI_ITERATIONS; ++i) {

      // resample transition using epsilon_greedy
      transitions.clear();
      fill(gen, simulator,
           [&simulator, &epsilon_greedy_on_q](){return epsilon_greedy_on_q(*simulator);},
           std::back_inserter(transitions),
           NB_TRANSITIONS * MAX_SAMPLE_LENGTH , MAX_EPISODE_LENGTH);
      {
        std::ofstream local_file {std::string("transition_online_lspi_")+std::to_string(i)+".csv"};
        store_transitions( transitions, local_file );
      }

      auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
                      greedy_on_q,
                      GAMMA,
                      transitions.begin(), transitions.end());
      std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

      // We could simply update 'q' using the weights of 'next_q' using std::swap(q, next_q);
      // here we use a progressive change in weights/params
      // q.params <- alpha x q.params + (1 - alpha) x next_q.params
      // q.params are nupplet
      // TODO use range to do the following ?
      auto q_params_it = q.params->begin();
      auto next_q_params_it = next_q.params->begin();
      for ( ; q_params_it != q.params->end(); ) {
        auto new_w = ALPHA * (*q_params_it) + (1.0 - ALPHA) * (*next_q_params_it);
        *q_params_it = new_w;
        ++q_params_it;
        ++next_q_params_it;
      }

      {
        std::ofstream local_file {std::string("test_online_lspi_")+std::to_string(i)+".csv"};
        test_policy(gen, greedy_on_q,
                    local_file, global_trace, i);
        std::ofstream qval_file {std::string("qval_online_lspi_")+std::to_string(i)+".csv"};
        sample_V_and_pi(q, qval_file, Q_SAMPLE_DENSITY);
      }

      // descrease epsilon
      epsilon *= 1.0 / EPSILON_TAU;
      epsilon = std::max( epsilon, EPSILON_THRESHOLD );
  }
  } // end of hard setting

  return 0;
}

// *****************************************************************************
// Helper functions to trace activity ******************************************

/** Run episods and trace results in file_csv
 *
 * traces globaly : iteration_nb episode_nb length
 * traces localy : episode_nb t s.pos s.vel a next_s.pos next_s.vel r
 */
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                 const POLICY& policy,
                 std::ostream& file_csv,
                 utils::trace::csv<>& g_trace, int ite_nb)
{
  auto sim = gdyn::problem::mountain_car::make();
  mc_system simulator {sim}; // simulator is sim, but handling discrete actions.

  // trace with header
  utils::trace::csv<> test {file_csv};
  std::array<std::string, 8> header {"## ep", "t", "s.pos", "s.vel", "a", "next_s.pos", "next_s.vel", "r"};
  test += header;

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    int length = 0;
    simulator = gdyn::problem::mountain_car::random_state(gen, gdyn::problem::mountain_car::parameters());
    for(auto [s, a, r, next_s, next_a]
          : gdyn::views::controller(simulator, policy)
          | gdyn::views::orbit(simulator)
          | rl2::views::sarsa
          | std::views::take(MAX_TEST_EPISODE_LENGTH)) {
      std::array<double, 8> line  { static_cast<double>(episode),
                                    static_cast<double>(length),
                                    s.position,
                                    s.velocity,
                                    static_cast<double>(a),
                                    next_s.position,
                                    next_s.velocity,
                                    r };
      test += line;
      
      ++length;
    }

    std::array<int, 3> line { ite_nb, episode, length };
    g_trace += line;
    
    if(length < (MAX_TEST_EPISODE_LENGTH))
      nb_success ++;
    sum_length += length;
  }
  std::cout << std::endl
            << "Success on " << int(100*nb_success/NB_TEST_EPISODES+.5) << "% of the episodes," << std::endl
            << "           (average episode length is " << sum_length/NB_TEST_EPISODES << ")." << std::endl;

}

/** Sample QValues and Best action.
 * samples are taken on a regular grid of size nb_bins x nb_bin on entire state space.
 *
 * traces localy : s.pos s.vel qval a_best
 */
template<typename Q>
void sample_V_and_pi(const Q& q,
                     std::ostream& file_csv,
                     int nb_bins)
{
  // trace with header
  utils::trace::csv<> traceqval {file_csv};
  std::array<std::string, 4> header {"## s.pos", "s.vel", "qval", "a_best"};
  traceqval += header;

  // sample uniformly the state
  auto [pos_min, pos_max] = std::make_tuple( -1.2, 0.6 );
  auto [vel_min, vel_max] = std::make_tuple( -0.07, 0.07 );

  auto greedy_on_q = rl2::discrete::greedy_ify(q); // or rl2::discrete::argmax_ify(q)

  double pos = 0.0;
  double vel = 0.0;
  for(int i = 0; i < nb_bins; ++i) {
    pos = rl2::enumerable::utils::digitize::to_value(i, pos_min, pos_max, nb_bins);
    for(int j = 0; j < nb_bins; ++j) {
      vel = rl2::enumerable::utils::digitize::to_value(j, vel_min, vel_max, nb_bins);

      // Get value of best action, and best action as int
      mc_S s{pos, vel};
      auto qval = q( s, greedy_on_q(s));
      int id_best_a = static_cast<std::size_t>(greedy_on_q(s));

      std::array<double, 4> line {
        pos,
        vel,
        qval,
        static_cast<double>(id_best_a) };
      traceqval += line;
    }
  }
}

/** Traces all the transistions
 *
 * trace localy : s.pos s.vel a next_s.pos next_s.vel
 */
void store_transitions(const std::vector<rl2::sarsa<mc_S, mc_A>>& transitions,
                       std::ostream& file_csv )
{
  // trace with header
  utils::trace::csv<> trace {file_csv};
  std::array<std::string, 5> header {"## s.pos", "s.vel", "a", "next_s.pos", "next_s.vel"};
  trace += header;

  for (auto t : transitions ) {
      std::array<double, 5> line {
        t.s.position,
        t.s.velocity,
        static_cast<double>(t.a),
        t.ss.position,
        t.ss.velocity };
      trace += line;
  }
}
