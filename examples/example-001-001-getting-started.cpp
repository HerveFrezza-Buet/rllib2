#include <iostream>
#include <iomanip>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read this file.
#include "weakest-link-problem.hpp"

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  auto environment = weakest_link::build_mdp(gen,
					     .75,   // Probability of answering correctly.
					     true); // true <=> display the reward table.
  environment = 'A';
  double total_gain = 0;

  // The call rl2::discrete::uniform<weakest_link::A>(gen)
  // provides a function f such as f() gives a random action.
  for(auto [s, a, r, ss, aa]
	: gdyn::ranges::tick(rl2::discrete::uniform<weakest_link::A>(gen)) // Feed the pipeline with random actions.
	| gdyn::views::orbit(environment)                                  // Drive the environment from these actions.
	| rl2::views::sarsa                                                // Collect (s, a, r, s', [a']) transitions.
	| std::views::take(30)) {                                          // Stop after 30 steps.
    total_gain += r;
    std::cout << static_cast<weakest_link::S::base_type>(s) << " : ";
    if(a)
      std::cout << "bank  --> " << r << "$." << std::endl;
    else {
      std::cout << "answer... ";
      if     (static_cast<weakest_link::S::base_type>(s)  == 'J') std::cout << "answer does not matter!";
      else if(static_cast<weakest_link::S::base_type>(ss) == 'A') std::cout << "bad answer.";
      else                                                        std::cout << "correct!";
      std::cout << std::endl;
    }
  }
  std::cout << "--------------------" << std::endl
	    << "Total gain: " << total_gain << "$." << std::endl
	    << std::endl;
  
  return 0;
}
