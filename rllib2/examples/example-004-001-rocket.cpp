#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <array>
#include <random>
#include <algorithm>
#include <cstddef>

#include <gdyn.hpp>
#include <rllib2.hpp>

#include "discrete-rocket-problem.hpp"
#include "my_rocket_config.hpp"

#define NB_PASSES 1000
#define GAMMA .99
#define ALPHA .05

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Let us build up the encapsulations of our rocket.
  auto params = make_params();
  auto rocket = types::base_continuous_system(params);
  auto relative_rocket = types::continuous_system(rocket, [target = params.ceiling_height/2](){return target;});
  auto exposed_rocket = types::exposed_system(relative_rocket);

  // We apply Q-learning to get the best controller, using a simple
  // tabular Q function.
  std::array<double, types::SA::size()> values;
  auto Q = rl2::enumerable::make_two_args_tabular<types::S, types::A>(values.begin());

  // The idea here is to explore discrete initial SxA situations
  // systematically. Just for fun, we explore them several time with a
  // random permutation.
  std::array<std::size_t, types::SA::size()> permutation;
  std::size_t idx = 0;
  for(auto& i : permutation) i = idx++;

  std::cout << std::endl << std::endl
	    << "Computing Q-learning passes:" << std::endl;
  for(std::size_t pass = 0; pass < NB_PASSES; ++pass) {
    std::cout << std::setw(5) << pass+1 << '/' << NB_PASSES << "\r     " << std::flush;
    std::shuffle(permutation.begin(), permutation.end(), gen);
    for(auto [init_state, command]
	  : permutation
	  | std::views::transform([](auto sa_idx) {types::SA::iterator it {sa_idx}; return *it;})) {
      exposed_rocket = init_state;
      auto reward = exposed_rocket(command);
      auto next_state = *exposed_rocket;
      
      // Nota : we do not need discrete_rocket simulator since making transition
      // will do the cast into dicrete states and actions.
      rl2::sarsa<types::S, types::A> transition {init_state, command, reward, next_state};

      // We apply the Q-learning update.
      double td_error = rl2::critic::td::error(Q, GAMMA, transition,
					       rl2::critic::td::enumerable::action::bellman::optimality<types::S, types::A, decltype(Q)>);
      rl2::critic::td::update(Q, transition.s, transition.a, ALPHA, td_error);
    }
  }
  std::cout << "Done.                   " << std::endl;


  {
    std::string filename {"rocket-discrete-controller.dat"};
    std::ofstream file {filename};

    // Now we have optimal Q, let us get the optimal policy and save it.
    auto controller = rl2::enumerable::greedy_ify(Q);
    
    // Let us save this policy as a dataset for further regression.
    for(auto s_it = types::S::begin(); s_it != types::S::end(); ++s_it) {
      auto s = *s_it;

      // s is casted into a discrete state when passed to the
      // controller, we get a discrete actionb from which we retrieve
      // the actual thrust, thanks to the static_cast.
      auto a = static_cast<types::A::base_type>(controller(s));
      file << s.error << ' ' << s.speed << ' ' << a.value;
      for(auto a_it = types::A::begin(); a_it != types::A::end(); ++a_it)
	file <<  ' ' << Q(s, a_it);
      file << ' ' << (Q(s, 0) - Q(s, 1)) << std::endl;
      file << std::endl;
    }

    std::cout << "File " << filename << " generated." << std::endl;
  }

  {
    std::string filename {"rocket-discrete-controller.plot"};
    std::ofstream file {filename};

    file << "set xlabel 'error'" << std::endl
	 << "set ylabel 'speed'" << std::endl
	 << "set zlabel 'thrust'" << std::endl
	 << "set title  'best discrete rocket controller'" << std::endl
	 << "splot 'rocket-discrete-controller.dat' using 1:2:3 with points pt 7 ps .5 notitle" << std::endl;

    std::cout << "File " << filename << " generated." << std::endl
	      << std::endl
	      << "Run : gnuplot -p " << filename << std::endl;
  }
  
  for(auto a_it = types::A::begin(); a_it != types::A::end(); ++a_it) {
    std::ostringstream filename;
    filename << "rocket-discrete-Q-A" << static_cast<std::size_t>(a_it) << ".plot";
    std::ofstream file {filename.str()};
    auto a = *a_it;

    file << "set xlabel 'error'" << std::endl
	 << "set ylabel 'speed'" << std::endl
	 << "set zlabel 'Q'" << std::endl
	 << "set title  'Q(s, a = " << a.value << ")'" << std::endl
	 << "splot 'rocket-discrete-controller.dat' using 1:2:" << static_cast<std::size_t>(a_it)+4 << " with points pt 7 ps .5 notitle" << std::endl;

    std::cout << "Run : gnuplot -p " << filename.str() << std::endl;
  }
			
  std::cout << std::endl
	    << std::endl;					   
    

  return 0;
}

  
