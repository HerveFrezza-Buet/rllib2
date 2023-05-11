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


// Let us define here some "quality" enumerable type.
struct quality_index_convertor {
  static std::string to  (std::size_t          index) {
    switch(index) {
    case 0:  return "bad"; 
    case 1:  return "neutral";
    default: return "good";
    }
  }
  static std::size_t from(const std::string& value) {
    if(value == "bad") return 0;
    if(value == "neutral") return 1;
    return 2;
  }
};

using quality = rl2::enumerable::count<std::string, 3, quality_index_convertor>; 


int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  // Let us initialize an environment.
  auto environment = weakest_link::build_mdp(gen, .75);
  environment = 'A';

  

  //////
  //
  // Tabular function and epsilon-ization.
  //
  //////

  
  {
    std::cout << std::endl
	      << "Tabular function and epsilon-ization" << std::endl
	      << "------------------------------------" << std::endl
	      << std::endl;
    
    // Let us defined a tabular function, associating a quality to
    // each state... 

    std::array<quality, weakest_link::S::size> values;
    std::size_t quality_idx = 0;
    for(auto& value : values) value = (quality_idx++ % quality::size);
    
    // We can make a tabular function from values.
    auto f = rl2::tabular::make_function<weakest_link::S>(values.begin());
    
    // Let us display the association
    std::cout << "Tabular quality function:" << std::endl;
    for(auto it = weakest_link::S::begin; it != weakest_link::S::end; ++it) {
      weakest_link::S state(it);
      std::cout << "  f(" << static_cast<weakest_link::S::base_type>(state)
		<< ") = " << static_cast<quality::base_type>(f(state))
		<< std::endl;
    }
      
    
    // We can make f epsilon-greedy (taking here epsilon as a
    // reference so that is can be changed afterwards.
    double epsilon; // The values will be set later in the loop.
    auto epsilon_f = rl2::discrete::epsilon(f, std::cref(epsilon), gen);

    // Let us count how many times f and epsilon_f differ.
    std::size_t nb_trials = 100000;
    for(auto eps : {.25, .5, .75}) {
      epsilon = eps; // epsilon_f depends on this setting, thanks to std::cref(epsilon).
      std::size_t nb_mismatches = 0;
      for([[maybe_unused]] const auto& unused
			     : gdyn::ranges::tick(rl2::discrete::uniform<weakest_link::S>(gen))
			     | std::views::take(nb_trials)
			     | std::views::filter([&f, &epsilon_f](const auto& s){return f(s) != epsilon_f(s);}))
	++nb_mismatches;
      std::cout.precision(2);
      std::cout << "epsilon = " << std::fixed << epsilon << ", mismatch_ratio = " << 100*(nb_mismatches/(double)nb_trials)
		<< "% (should be close to " << 100*epsilon * (quality::size - 1) / (double)(quality::size)
		<< "%)." << std::endl;
    }
  }

  //////
  //
  // Greedy-ness.
  //
  //////
  
  {
    std::cout << std::endl
	      << "Greedy-ness" << std::endl
	      << "-----------" << std::endl
	      << std::endl;
    
    // Let us get an array of random values for each state.
    std::array<double, weakest_link::S::size> values;
    
    // Let us build a tabular value function from it.
    auto V        = rl2::tabular::make_function<weakest_link::S>(values.begin());
    auto greedy_V = rl2::discrete::greedy<weakest_link::S>(V);

    // Note that V and greedy_V are created once, beforehand. They
    // depend on the future content of the values array.

    for(unsigned int i = 0; i< 5; ++i) {
      // We (re)initialize the values randomly
      for(auto& value : values) value = std::uniform_real_distribution<double>(0, 1)(gen);
      // Let us display the new association
      std::cout << "Value function: [";
      for(auto it = weakest_link::S::begin; it != weakest_link::S::end; ++it) {
	weakest_link::S state(it);
	if(it != weakest_link::S::begin) std::cout << ", ";
	std::cout << static_cast<weakest_link::S::base_type>(state) << ": " <<  V(state);
      }
      weakest_link::S argmax = greedy_V();
      std::cout << "] : argmax = " << static_cast<weakest_link::S::base_type>(argmax) << std::endl;
    }
  }
  

  
  return 0;
}
