#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <functional>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read this file.
#include "weakest-link-problem.hpp"


int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  // Let us initialize an environment.
  auto environment = weakest_link::build_mdp(gen, .75);
  environment = 'A';

  {
    // Let us defined a tabular function, associating a state to each
    // state... let us implement the identity here.

    rl2::tabular::function<weakest_link::S, weakest_link::S> f {};
    
    for(auto it = weakest_link::S::begin; it != weakest_link::S::end; ++it) f(it) = it;

    // We can make f epsilon-greedy (taking here epsilon as a
    // reference so that is can be changed afterwards.
    double epsilon = .2;
    auto epsilon_f = rl2::discrete::epsilon(f, std::cref(epsilon), gen);

    // Let us count how many times f and epsilon_f differ.
    std::size_t nb_trials     = 1000;
    std::size_t nb_mismatches = 0;
    for(auto s
	  : gdyn::ranges::tick(rl2::discrete::uniform<weakest_link::S>(gen))
	  | std::views::take(nb_trials))
      if(f(s) != epsilon_f(s)) nb_mismatches += 1;
    std::cout << "% mismatch = " << (nb_mismatches/(double)nb_trials)
	      << " (should be " << epsilon * (weakest_link::S::size - 1)
	      << ")" << std::endl;
      
  }

  

  
  return 0;
}
