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
  static std::string to(std::size_t index) {
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
  
  std::cout.precision(2);
  std::cout << std::boolalpha << std::fixed;
  
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
    
    // We can make a tabular function from the values.
    // TODO donc, f est une fonction de S -> quality
    auto f = rl2::tabular::make_function<weakest_link::S>(values.begin());
    
    // Let us display the association
    std::cout << "Tabular quality function:" << std::endl;
    for(auto it = weakest_link::S::begin; it != weakest_link::S::end; ++it) {
      // TODO et pourquoi pas state{it} ??
      weakest_link::S state(it);
      std::cout << "  f(" << static_cast<weakest_link::S::base_type>(state   )
		<< ") = " << static_cast<quality::base_type>        (f(state))
		<< std::endl;
    }
      
    // We can make f epsilon-greedy (taking here epsilon as a
    // reference with std::cref, so that is can be changed afterwards.
    double epsilon; // The values will be set later in the loop.
    // TODO epsilon_sampler ??
    // TODO connait pas 'cref'
    auto epsilon_f = rl2::discrete::epsilon(f, std::cref(epsilon), gen);

    // Let us count how many times f and epsilon_f differ.
    
    std::size_t nb_trials = 100000;
    for(auto eps : {.25, .5, .75}) {
      // TODO more explicit ? like f.epsilon = eps OR f.set_epsilon(eps) ??
      epsilon = eps; // epsilon_f depends on this setting, thanks to std::cref(epsilon).
      std::size_t nb_mismatches = 0;
      for([[maybe_unused]] const auto& unused
			     : gdyn::ranges::tick(rl2::discrete::uniform<weakest_link::S>(gen))
			     | std::views::take(nb_trials)
			     | std::views::filter([&f, &epsilon_f](const auto& s){return f(s) != epsilon_f(s);}))
	++nb_mismatches;
      std::cout << "epsilon = " << epsilon << ", mismatch_ratio = " << 100*(nb_mismatches/(double)nb_trials)
		<< "% (should be close to " << 100*epsilon * (quality::size - 1) / (double)(quality::size)
		<< "%)." << std::endl;
    }
  }

  
  //////
  //
  // Tabular table.
  //
  //////

  {
    std::cout << std::endl
	      << "Tabular table" << std::endl
	      << "-------------" << std::endl
	      << std::endl;

    std::array<double, weakest_link::SA::size> values;
    for(auto& value : values) value = std::uniform_real_distribution(0., 1.)(gen);
    
    auto Q = rl2::tabular::make_table<weakest_link::S, weakest_link::A>(values.begin());

    // From this table, we can build up a greedy policy, since weakest_link::A is enumerable (this is mandatory for internal argmax);
    auto greedy_on_Q = rl2::discrete::greedy(Q); // or rl2::discrete::argmax(Q)

    // We can espilon-ize such things to get an epsilon greedy policy.
    double epsilon = .2;
    auto epsilon_greedy_on_Q = rl2::discrete::epsilon(rl2::discrete::argmax(Q), epsilon, gen);

    // Let us display the values.
    
    std::cout << "Q |";
    for(auto it = weakest_link::A::begin; it != weakest_link::A::end; ++it) std::cout << ' ' << std::setw(5) << static_cast<weakest_link::A::base_type>(it);
    std::cout << std::endl;

    std::cout << "--+";
    for(auto it = weakest_link::A::begin; it != weakest_link::A::end; ++it) std::cout << std::string(6, '-');
    std::cout << std::endl;
    
    for(auto s_it = weakest_link::S::begin; s_it != weakest_link::S::end; ++s_it) {
      weakest_link::S s(s_it);
      std::cout << static_cast<weakest_link::S::base_type>(s) << " |";
      auto QS = Q(s); // Q(S) is a function taking actions (i.e. weakest_link::A) as arguments and returning values.
      for(auto a_it = weakest_link::A::begin; a_it != weakest_link::A::end; ++a_it)
	std::cout << ' ' << std::setw(5) <<  QS(a_it); // a_it is implicitly converted into weakest_link::A here.
      weakest_link::A best = greedy_on_Q(s);
      std::cout << "  -->  argmax = " << static_cast<weakest_link::A::base_type>(best) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Epsilon greedy: " << std::endl;
    weakest_link::S s {'H'};
    std::cout << static_cast<weakest_link::S::base_type>(s) << ": ";
    unsigned int nb_trues  = 0;
    unsigned int nb_trials = 1000;
    for(unsigned int i = 0; i < nb_trials; ++i)
      if(static_cast<weakest_link::A::base_type>(epsilon_greedy_on_Q(s))) ++nb_trues;
    std::cout << 100*nb_trues/(double)nb_trials << "% are the 'true' action." << std::endl;
    std::cout << std::endl;
    
  }

  
  return 0;
}
