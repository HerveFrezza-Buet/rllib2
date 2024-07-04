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

#include <Eigen/Dense>

#include <rllib2.hpp>
#include <rllib2-eigen.hpp>

#include <gdyn.hpp>

// Log results in traces
#include <utils/trace.hpp>

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

template<typename RANDOM_DEVICE, typename POLICY>
void test_policy_with_trace(RANDOM_DEVICE& gen,
                            const POLICY& policy,
                            const std::string& trace_name,
                            utils::Trace& g_trace, int ite_nb);

template<typename Q>
void store_weights(const Q& q,
                   utils::Trace& trace,
                   int current_it);

template<typename Q>
void sample_V_and_pi(const Q& q,
                     const std::string& trace_name,
                     int nb_bins);

void store_transitions(const std::vector<rl2::sarsa<S, A>>& transitions,
                       const std::string& trace_name );

// *****************************************************************************
//                                                                          MAIN
// *****************************************************************************
int main(int argc, char *argv[])
{

  // TODO with arguments : nb_trans, nb_iter, alpha, tau_epsilon
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
  store_transitions( transitions,
                     "test_transitions_start");

  // For LSPI, we need a parametrized Q function, and related policies.
  Q q      {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};
  Q next_q {std::make_shared<s_feature>(make_state_feature()), std::make_shared<params>()};

  // raw_Q q      {std::make_shared<raw_s_feature>(), std::make_shared<raw_params>()};
  // raw_Q next_q {std::make_shared<raw_s_feature>(), std::make_shared<raw_params>()};
  // // TODO j'ai eu du mal à retrouver la def rl2::linear::discrete_a::q
  // // (pour q.s_feature)
  // q.s_feature->data = std::make_shared<std::array<double, raw_s_feature::dim>>();
  // next_q.s_feature->data = std::make_shared<std::array<double, raw_s_feature::dim>>();


  auto   greedy_on_q         = rl2::discrete::greedy_ify(q);
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);

  std::cout << "Q is a linear combination of RBFs" << std::endl;
  std::cout << "   - with continuous state space and " << A::size() << " discrete actions" << std::endl;
  std::cout << "   - " << q.s_feature->dim << " features"<< std::endl;
  std::cout << "   - " << q.params->dim << " parameters" << std::endl;


  // Trace to log test statistics
  std::stringstream header_len;
  header_len << "it\tep\tlen";
  utils::Trace global_trace; // log of Q perf
  global_trace.add_header( header_len.str() );
  utils::Trace next_trace; // lof of next_Q perd
  next_trace.add_header( header_len.str() );

  // Trace for weights
  std::stringstream header_w;
  header_w << "it";
  for (unsigned int i = 0; i < q.params->dim; ++i) {
    header_w << "\tw_" << i;
  }
  utils::Trace w_trace; // log of Q params
  w_trace.add_header( header_w.str() );


  test_policy_with_trace(gen, rl2::discrete_a::random_policy<S, A>(gen),
                         "test_global_random", global_trace, -1);
  test_policy_with_trace(gen, rl2::discrete_a::random_policy<S, A>(gen),
                         "test_next_random", next_trace, -1);

  // Let us initialize Q from the random policy. (true means that we
  // want to actually compute the error. 0 is returned with false).
  double gamma = .95;
  double error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
						      rl2::discrete_a::random_policy<S, A>(gen),
						      gamma,
						      transitions.begin(), transitions.end());
  std::cout << "LSTD error of the random policy = " << error << std::endl;

  test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
                         "test_next_init", next_trace, -1);
  // TODO copy weights from next_q  to q can be achieved with a swap
  std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.
  // test_policies
  test_policy_with_trace(gen, rl2::discrete::greedy_ify(q),
                         "test_global_init", global_trace, -1);

  store_weights(q, w_trace, -1);

  sample_V_and_pi(q, "test_qval_init", 50);

  // LSPI iteration
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 0; i < nb_lspi_iterations; ++i) {
    // TODO copy weights from next_q  to q can be achieved with a swap
    // std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.


    // TODO debug
    // std::cout << "AVANT LSPI - q.params" << std::endl;
    // std::cout << *(q.params) << std::endl;

    if (uniform_sampling) {
      // TODO no need to re-sample the transitions set, as they can be re-used.
    }
    else {
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

      store_transitions( transitions,
                         "test_transitions_"+std::to_string(i));
    }

    auto error = rl2::eigen::critic::discrete_a::lstd<true>(next_q,
							     // epsilon_greedy_on_q,
							    greedy_on_q,
							    gamma,
							    transitions.begin(), transitions.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    // std::swap(q, next_q); // swapping q functions swaps their contents, which are share pointers.
    // TODO but it is also possible to use a progressive change in params
    // q.params <- alpha x q.params + (1 - alpha) x next_q.params
    // q.params are nupplet
    // TODO use range to do the following ?
    auto q_params_it = q.params->begin();
    auto next_q_params_it = next_q.params->begin();
    // TODO double delta_weight {0.0};
    for ( ; q_params_it != q.params->end(); ) {
      auto new_w = alpha * (*q_params_it) + (1.0 - alpha) * (*next_q_params_it);
      // delta_weight += std::abs((*q_params_it) - new_w);
      *q_params_it = new_w;
      ++q_params_it;
      ++next_q_params_it;
    }

    store_weights(q, w_trace, i);

    // TODO debug
    // std::cout << "APRES LSPI - q.params" << std::endl;
    // std::cout << *(q.params) << std::endl;

    test_policy_with_trace(gen, rl2::discrete::greedy_ify(q),
                           "test_global_lspi_"+std::to_string(i), global_trace, i);
    test_policy_with_trace(gen, rl2::discrete::greedy_ify(next_q),
                           "test_next_lspi_"+std::to_string(i), next_trace, i);
    sample_V_and_pi(q, "test_qval_lspi_"+std::to_string(i), 50 );

    epsilon *= 1.0 / tau_epsilon;
    epsilon = std::max( epsilon, epsilon_threshold );
  }

  std::cout << "Saving global_trace in lerning_xxx.csv" << std::endl;
  std::ofstream outfile_q( "learning_global.csv" );
  utils::Trace::write( outfile_q, global_trace );
  outfile_q.close();
  std::ofstream outfile_n( "learning_next.csv" );
  utils::Trace::write( outfile_n, next_trace );
  outfile_n.close();

  std::cout << "Saving weights.csv" << std::endl;
  std::ofstream outfile_w( "weights.csv" );
  utils::Trace::write( outfile_w, w_trace );
  outfile_w.close();

  // test_policy(gen, next_q);
  // TODO run_policy(S(), next_q);
  
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
  auto sim = gdyn::problem::mountain_car::make();
  mountain_car simulator {sim}; // simulator is sim, but handling discrete actions.

  // trace with header
  std::stringstream header;
  header << "ep\tt\ts.pos\ts.vel\ta";
  header << "\tnext_s.pos\tnext_s.vel\tr";
  utils::Trace local_trace;
  local_trace.add_header( header.str() );

  unsigned int nb_success = 0;
  unsigned int sum_length = 0;
  for(unsigned int episode = 0; episode < NB_TEST_EPISODES; ++episode) {
    std::cout << "episode " << std::setw(6) << episode + 1 << "/" << NB_TEST_EPISODES << "  \r" << std::flush;
    unsigned int length = 0;
    simulator = gdyn::problem::mountain_car::random_state(gen, gdyn::problem::mountain_car::parameters());
    for(auto [s, a, r, next_s, next_a]
          : gdyn::views::controller(simulator, policy)
          | gdyn::views::orbit(simulator)
          | rl2::views::sarsa
          | std::views::take(MAX_TEST_EPISODE_LENGTH)) {

      local_trace.push_to_state( static_cast<double>(episode) );
      local_trace.push_to_state( static_cast<double>(length) );
      local_trace.push_to_state( s.position );
      local_trace.push_to_state( s.velocity );
      local_trace.push_to_state( static_cast<double>(a) );
      local_trace.push_to_state( next_s.position );
      local_trace.push_to_state( next_s.velocity );
      local_trace.push_to_state( r );
      local_trace.store_state();

      ++length;

    }
    g_trace.push_to_state( static_cast<double>(ite_nb) );
    g_trace.push_to_state( static_cast<double>(episode) );
    g_trace.push_to_state( static_cast<double>(length) );
    g_trace.store_state();

    if(length < (MAX_TEST_EPISODE_LENGTH))
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

template<typename Q>
void store_weights(const Q& q,
                   utils::Trace& trace,
                   int current_it)

{
  trace.push_to_state(static_cast<double>(current_it));
  for( auto w: *(q.params)) {
    trace.push_to_state( w );
  }
  trace.store_state();
}

template<typename Q>
void sample_V_and_pi(const Q& q,
                     const std::string& trace_name,
                     int nb_bins)
{
  // trace with header
  std::stringstream header;
  header << "pos\tvel\tqval\ta_best";
  utils::Trace local_trace;
  local_trace.add_header( header.str() );

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
      S s{pos, vel};
      auto qval = q( s, greedy_on_q(s));
      int id_best_a = static_cast<std::size_t>(greedy_on_q(s));

      local_trace.push_to_state( pos );
      local_trace.push_to_state( vel );
      local_trace.push_to_state( qval );
      local_trace.push_to_state( static_cast<double>(id_best_a) );
      local_trace.store_state();
    }
  }

  std::cout << "Saving trace in " << trace_name << ".csv" << std::endl;
  std::ofstream outfile( trace_name+".csv" );
  utils::Trace::write( outfile, local_trace );
  outfile.close();
}

void store_transitions(const std::vector<rl2::sarsa<S, A>>& transitions,
                       const std::string& trace_name )
{
    // for(auto [s, a, r, next_s, next_a]
    //       : gdyn::views::controller(simulator, policy)
    //       | gdyn::views::orbit(simulator)
    //       | rl2::views::sarsa
    //       | std::views::take(MAX_TEST_EPISODE_LENGTH)) {

  // trace with header
  std::stringstream header;
  header << "pos\tvel\tact\tn_pos\tn_vel";
  utils::Trace local_trace;
  local_trace.add_header( header.str() );

  for (auto t : transitions ) {
    local_trace.push_to_state( t.s.position );
    local_trace.push_to_state( t.s.velocity );
    local_trace.push_to_state( static_cast<double>(t.a) );
    local_trace.push_to_state( t.ss.position );
    local_trace.push_to_state( t.ss.velocity );
    local_trace.push_to_state( t.r );
    local_trace.store_state();
  }

  std::cout << "Saving transitions in " << trace_name << ".csv" << std::endl;
  std::ofstream outfile( trace_name+".csv" );
  utils::Trace::write( outfile, local_trace );
  outfile.close();
}
