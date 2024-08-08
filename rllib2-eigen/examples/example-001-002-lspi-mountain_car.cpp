// TODO uniform sampling of the problem
// example-001-002-lspi-mountain_car 1000 40 0.0 20.0
//
// R> run_mountain( nb_trans=1000, nb_lspi=40, tau_epsilon=20, alpha=0.0)
// R> make_plots( "mountain", 39 )

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

#define EPSILON_START          1.0
#define EPSILON_TAU            0.1
#define EPSILON_THRESHOLD      0.0001

#define ALPHA                  0.05

template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                   const POLICY& policy,
                   std::ostream& test_csv,
                   utils::trace::csv<>& g_trace, int ite_nb);

// template<typename Q>
// void sample_V_and_pi(const Q& q,
//                      const std::string& trace_name,
//                      int nb_bins);

// void store_transitions(const std::vector<rl2::sarsa<S, A>>& transitions,
//                        const std::string& trace_name );

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
  
  // First, we fill the dataset with a random policy
  std::cout << "Filling the dataset with " << NB_TRANSITIONS << "..." << std::flush;
  // applied on 'nb_transition' uniformly sampled starting points
  fill_uniform(gen, simulator,
               rl2::discrete::uniform_sampler<mc_A>(gen),
               std::back_inserter(transitions),
               NB_TRANSITIONS, MAX_SAMPLE_LENGTH);
// else {
//   // applied on 'nb_transition' starting points chosen according to problem
  //   fill(gen, simulator,
  //        rl2::discrete::uniform_sampler<A>(gen),
  //        std::back_inserter(transitions),
  //        nb_transitions, MAX_EPISODE_LENGTH);
  // }
  std::cout << " got " << transitions.size() << " samples, ("
            << std::ranges::count_if(transitions, [](auto& transition){return transition.is_terminal();})
            << " are terminal transitions)." << std::endl;
  
  // For LSPI, we need a parametrized Q function, and related policies.
  mc_Q q      {std::make_shared<mc_s_feature>(utils::mountain_car::make_state_feature()), std::make_shared<mc_params>()};
  mc_Q next_q {std::make_shared<mc_s_feature>(utils::mountain_car::make_state_feature()), std::make_shared<mc_params>()};

  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << mc_A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " feature"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;
  
  // utils::Trace w_trace; // log of Q params
  // w_trace.add_header( header_w.str() );

std::ofstream global_file {"global.csv"};
utils::trace::csv<> global_trace {global_file}; 

std::array<std::string, 3> header {"nb", "ep", "len"};
global_trace += header;

{
  
    std::ofstream local_file {"test_global_random.csv"};
    test_policy(gen, rl2::discrete_a::random_policy<mc_S, mc_A>(gen),
                local_file, global_trace, -1);
  }
  
  // {
  // std::ofstream local_file {"test_next_random.csv"};
  // test_policy(gen, rl2::discrete_a::random_policy<S, A>(gen),
  //             local_file, next_trace, -1);
  // }
  
//   // Let us initialize Q from the random policy. (true means that we
//   // want to actually compute the error. 0 is returned with false).
//   double gamma = .95;
//   double error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
// 						      rl2::discrete_a::random_policy<S, A>(gen),
// 						      gamma,
// 						      transitions.begin(), transitions.end());
//   std::cout << "LSTD error of the random policy = " << error << std::endl;

//   test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
//                          "test_next_init", next_trace, -1);
//   // TODO copy weights from next_q  to q can be achieved with a swap
//   std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.
//   // test_policies
//   test_policy_with_trace(gen, rl2::discrete::greedy_ify(q),
//                          "test_global_init", global_trace, -1);

//   store_weights(q, w_trace, -1);

//   sample_V_and_pi(q, "test_qval_init", 50);

//   // LSPI iteration
//   std::cout << "Training using LSPI" << std::endl;
//   for(unsigned int i = 0; i < nb_lspi_iterations; ++i) {
//     // TODO copy weights from next_q  to q can be achieved with a swap
//     // std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.


//     // TODO debug
//     // std::cout << "AVANT LSPI - q.params" << std::endl;
//     // std::cout << *(q.params) << std::endl;

//     if (uniform_sampling) {
//       // TODO no need to re-sample the transitions set, as they can be re-used.
//     }
//     else {
//       // sample using random policy AND epsilon_greedy
//       transitions.clear();
//       fill(gen, simulator,
//            rl2::discrete::uniform_sampler<A>(gen),
//            std::back_inserter(transitions),
//            nb_transitions/2, MAX_EPISODE_LENGTH);
//       fill(gen, simulator,
//            [&simulator, &epsilon_greedy_on_q](){return epsilon_greedy_on_q(*simulator);},
//            std::back_inserter(transitions),
//            nb_transitions/2, MAX_EPISODE_LENGTH);

//       store_transitions( transitions,
//                          "test_transitions_"+std::to_string(i));
//     }

//     auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
// 							     // epsilon_greedy_on_q,
// 							    greedy_on_q,
// 							    gamma,
// 							    transitions.begin(), transitions.end());
//     std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

//     // std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.
//     // TODO but it is also possible to use a progressive change in params
//     // q.params <- alpha x q.params + (1 - alpha) x next_q.params
//     // q.params are nupplet
//     // TODO use range to do the following ?
//     auto q_params_it = q.params->begin();
//     auto next_q_params_it = next_q.params->begin();
//     // TODO double delta_weight {0.0};
//     for ( ; q_params_it != q.params->end(); ) {
//       auto new_w = alpha * (*q_params_it) + (1.0 - alpha) * (*next_q_params_it);
//       // delta_weight += std::abs((*q_params_it) - new_w);
//       *q_params_it = new_w;
//       ++q_params_it;
//       ++next_q_params_it;
//     }

//     store_weights(q, w_trace, i);

//     // TODO debug
//     // std::cout << "APRES LSPI - q.params" << std::endl;
//     // std::cout << *(q.params) << std::endl;

//     test_policy_with_trace(gen, rl2::discrete::greedy_ify(q),
//                            "test_global_lspi_"+std::to_string(i), global_trace, i);
//     test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
//                            "test_next_lspi_"+std::to_string(i), next_trace, i);
//     sample_V_and_pi(q, "test_qval_lspi_"+std::to_string(i), 50 );

//     epsilon *= 1.0 / tau_epsilon;
//     epsilon = std::max( epsilon, epsilon_threshold );
//   }

// std::cout << "Saving global_trace in lerning_xxx.csv" << std::endl;
// std::ofstream outfile_q( "learning_global.csv" );
// utils::Trace::write( outfile_q, global_trace );
// outfile_q.close();
// std::ofstream outfile_n( "learning_next.csv" );
// utils::Trace::write( outfile_n, next_trace );
// outfile_n.close();

// std::cout << "Saving weights.csv" << std::endl;
// std::ofstream outfile_w( "weights.csv" );
// utils::Trace::write( outfile_w, w_trace );
// outfile_w.close();

  return 0;
}



/** Run episods and store results in file
 *
 * episode_nb t s.x s.x_dot s.theta s.theta_dot a next_s.x ... next_s.theta_dot r
 *
 * Returns: nb_success (int)
 */
template<typename RANDOM_DEVICE, typename POLICY>
void test_policy(RANDOM_DEVICE& gen,
                 const POLICY& policy,
                 std::ostream& test_csv,
                 utils::trace::csv<>& g_trace, int ite_nb)
{
  auto sim = gdyn::problem::mountain_car::make();
  mc_system simulator {sim}; // simulator is sim, but handling discrete actions.

  // trace with header
  utils::trace::csv<> test {test_csv};
  std::array<std::string, 8> header {"ep", "t", "s.pos", "s.vel", "a", "next_s.pos", "next_s.vel", "r"};
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

// template<typename Q>
// void store_weights(const Q& q,
//                    utils::Trace& trace,
//                    int current_it)

// {
//   trace.push_to_state(static_cast<double>(current_it));
//   for( auto w: *(q.params)) {
//     trace.push_to_state( w );
//   }
//   trace.store_state();
// }

// template<typename Q>
// void sample_V_and_pi(const Q& q,
//                      const std::string& trace_name,
//                      int nb_bins)
// {
//   // trace with header
//   std::stringstream header;
//   header << "pos\tvel\tqval\ta_best";
//   utils::Trace local_trace;
//   local_trace.add_header( header.str() );

//   // sample uniformly the state
//   auto [pos_min, pos_max] = std::make_tuple( -1.2, 0.6 );
//   auto [vel_min, vel_max] = std::make_tuple( -0.07, 0.07 );

//   auto greedy_on_q = rl2::discrete::greedy_ify(q); // or rl2::discrete::argmax_ify(q)

//   double pos = 0.0;
//   double vel = 0.0;
//   for(int i = 0; i < nb_bins; ++i) {
//     pos = rl2::enumerable::utils::digitize::to_value(i, pos_min, pos_max, nb_bins);
//     for(int j = 0; j < nb_bins; ++j) {
//       vel = rl2::enumerable::utils::digitize::to_value(j, vel_min, vel_max, nb_bins);

//       // Get value of best action, and best action as int
//       S s{pos, vel};
//       auto qval = q( s, greedy_on_q(s));
//       int id_best_a = static_cast<std::size_t>(greedy_on_q(s));

//       local_trace.push_to_state( pos );
//       local_trace.push_to_state( vel );
//       local_trace.push_to_state( qval );
//       local_trace.push_to_state( static_cast<double>(id_best_a) );
//       local_trace.store_state();
//     }
//   }

//   std::cout << "Saving trace in " << trace_name << ".csv" << std::endl;
//   std::ofstream outfile( trace_name+".csv" );
//   utils::Trace::write( outfile, local_trace );
//   outfile.close();
// }

// void store_transitions(const std::vector<rl2::sarsa<S, A>>& transitions,
//                        const std::string& trace_name )
// {
//     // for(auto [s, a, r, next_s, next_a]
//     //       : gdyn::views::controller(simulator, policy)
//     //       | gdyn::views::orbit(simulator)
//     //       | rl2::views::sarsa
//     //       | std::views::take(MAX_TEST_EPISODE_LENGTH)) {

//   // trace with header
//   std::stringstream header;
//   header << "pos\tvel\tact\tn_pos\tn_vel";
//   utils::Trace local_trace;
//   local_trace.add_header( header.str() );

//   for (auto t : transitions ) {
//     local_trace.push_to_state( t.s.position );
//     local_trace.push_to_state( t.s.velocity );
//     local_trace.push_to_state( static_cast<double>(t.a) );
//     local_trace.push_to_state( t.ss.position );
//     local_trace.push_to_state( t.ss.velocity );
//     local_trace.push_to_state( t.r );
//     local_trace.store_state();
//   }

//   std::cout << "Saving transitions in " << trace_name << ".csv" << std::endl;
//   std::ofstream outfile( trace_name+".csv" );
//   utils::Trace::write( outfile, local_trace );
//   outfile.close();
// }
