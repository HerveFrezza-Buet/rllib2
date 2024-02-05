#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <functional>
#include <string>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read this file.
#include "weakest-link-problem.hpp"

struct Params {
  // Few parameters
  double learning_rate = .05;
  double gamma         = .95;
  double epsilon       = .20;
  double correct_proba = .80;

  // Experimental setup
  std::size_t nb_epochs     = 1000;
  std::size_t epoch_length  =  100;
  std::size_t nb_test_steps = 1000;
};

template<typename QTABLE>
void display_greedy_policy(const QTABLE& Q) {
  // The questions for which the greedy policy says 'bank' are in upper case.
  std::cout << "  greedy: ";
  for(auto it = weakest_link::S::begin(); it != weakest_link::S::end(); ++it) {
    bool a = static_cast<weakest_link::A::base_type>(rl2::discrete::algo::argmax<weakest_link::A>(Q(weakest_link::S(it))));
    char c = *it;
    if(!a) c += 32;
    std::cout << c;
  }
}

template<bool is_sarsa, typename MDP, typename QTABLE, typename POLICY, typename RANDOM_GENERATOR>
void train(MDP& environment, QTABLE& q, const POLICY& exploration_policy, Params& params, RANDOM_GENERATOR& gen) {
  auto random_state_generator = rl2::discrete::uniform_sampler<weakest_link::S>(gen);
  for(unsigned int epoch=0; epoch < params.nb_epochs; ++epoch) {
    environment = random_state_generator(); // We implement exploring starts.
    for(auto transition
	  : gdyn::ranges::controller(environment, exploration_policy) 
	  | gdyn::views::orbit(environment)
	  | rl2::views::sarsa     
	  | std::views::take(params.epoch_length)) {
      if constexpr (is_sarsa) {
	auto bellman_op = rl2::critic::td::bellman::evaluation<typename MDP::state_type, typename MDP::command_type, QTABLE>;
	rl2::critic::td::update(q, transition.s, transition.a, params.learning_rate,
				rl2::critic::td::error(q, params.gamma, transition, bellman_op)); // SARSA
      }
      else {
	auto bellman_op = rl2::critic::td::discrete::bellman::optimality<typename MDP::state_type, typename MDP::command_type, QTABLE>;
	rl2::critic::td::update(q, transition.s, transition.a, params.learning_rate,
				rl2::critic::td::error(q, params.gamma, transition, bellman_op)); // Q-Learning
      }
      display_greedy_policy(q); std::cout << '\r' << std::flush;
    }

    params.epsilon *= .9;
    // we slowly decrease epsilon. As this is the variable taken by
    // reference (std::cref) when the epsilon greedy policy has been
    // created, this change will modify the policy.
  }
  display_greedy_policy(q); std::cout << std::endl;
}

template<typename MDP, typename POLICY>
void test(MDP& environment, POLICY& test_policy, Params& params) {
  double total_gain = 0;
  environment = 'A';
  for(auto [s, a, r, ss, aa]
	: gdyn::ranges::controller(environment, test_policy) 
	| gdyn::views::orbit(environment)                                 
	| rl2::views::sarsa                                               
	| std::views::take(params.nb_test_steps))                                    
    total_gain += r;
    
  std::cout << "  Got an average gain for " << weakest_link::S::size() << " questions: "
	    << weakest_link::S::size() * total_gain/(double)(params.nb_test_steps) << "$." << std::endl;
}

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  Params params;
  
  std::cout.precision(2);
  std::cout << std::boolalpha << std::fixed;
  
  // First we need parameters for the Q function.
  std::array<double, weakest_link::SA::size()> values;

  // This is the Q function.
  auto Q = rl2::tabular::make_two_args_function<weakest_link::S, weakest_link::A>(values.begin());

  // Let us define an epsilon and a epsilon-greedy policy.
  auto greedy_policy         = rl2::discrete::greedy_ify(Q);
  auto epsilon_greedy_policy = rl2::discrete::epsilon_ify(greedy_policy, std::cref(params.epsilon), gen);
  // The policy is sensitive to further modifications of params.epsilon, thanks to the use of std::cref.

  // This is the environment
  auto environment = weakest_link::make_mdp(gen, params.correct_proba);

  std::cout << "Player skill : " << int(100 * params.correct_proba) << "% of correct answers." << std::endl;

  std::cout << "Sarsa :" << std::endl;
  for(auto& value : values) value = std::uniform_real_distribution(0., 1.)(gen); // Let us initialize the values randomly
  train<true>(environment, Q, epsilon_greedy_policy, params, gen); // train<true> triggers sarsa learning.
  test(environment, greedy_policy, params);
  
  std::cout << "Q-Learning :" << std::endl;
  for(auto& value : values) value = std::uniform_real_distribution(0., 1.)(gen); // Let us initialize the values randomly
  train<false>(environment, Q, epsilon_greedy_policy, params, gen); // train<false> triggers Q-learning.
  test(environment, greedy_policy, params);


  return 0;
}

