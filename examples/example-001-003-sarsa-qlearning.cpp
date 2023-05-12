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


int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::cout.precision(2);
  std::cout << std::boolalpha << std::fixed;

  // Few parameters
  double learning_rate = .05;
  double gamma         = .95;
  double epsilon       = .10;
  
  // First we need parameters for the Q function.
  std::array<double, weakest_link::SA::size> values;

  // This is the Q function.
  auto Q = rl2::tabular::make_table<weakest_link::S, weakest_link::A>(values.begin());

  // Let us define an epsilon and a epsilon-greedy policy.
  auto greedy_policy         = rl2::discrete::greedy(Q);
  auto epsilon_greedy_policy = rl2::discrete::epsilon(greedy_policy, epsilon, gen);

  // This is the environment
  auto environment = weakest_link::build_mdp(gen, .75);

    
  std::size_t nb_steps_train = 10000;
  std::size_t nb_steps_test  = 100;
  
  // SARSA
  {
    // Let us initialize an environment.
    environment = 'A';

    // Let us initialize the values randomly
    for(auto& value : values) value = std::uniform_real_distribution(0., 1.)(gen);

    // // Let us run episodes and apply the sarsa learning rule. We use
    // // the epsilon-greedy to explore.
    // for(auto transition
    // 	  : gdyn::ranges::controller(environment, epsilon_greedy_policy) 
    // 	  | gdyn::views::orbit(environment)
    // 	  | rl2::views::sarsa     
    // 	  | std::views::take(nb_steps_train))
    //   rl2::critic::td::update(Q, transition.s, transition.a, learning_rate,
    // 			      rl2::critic::td::evaluation_error(Q, gamma, transition));

    // Let us count the reward, using the greedy policy
    double total_gain = 0;
    environment = 'A';
    for(auto [s, a, r, ss, aa]
	: gdyn::ranges::controller(environment, greedy_policy) 
	| gdyn::views::orbit(environment)                                 
	| rl2::views::sarsa                                               
	| std::views::take(nb_steps_test))                                    
      total_gain += r;

    std::cout << "Sarsa : got an average gain of " << total_gain/(double)nb_steps_test << "$." << std::endl;
  }

  // Q-Learning (the same, only the TD-error changes)
  {
  }


  return 0;
}

