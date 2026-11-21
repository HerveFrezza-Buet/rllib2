#include <iostream>
#include <iomanip>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read this file.
#include "weakest-link-problem.hpp"


// This tutorial show dynamic programming algorithms, for discrete
// states and actions. They can be used when the transition
// distributions and the reward function are available.

// All computations are based on tabular functions, whos values are
// stored in an array.


int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  // First, we need an array to store the values.
  std::array<double, weakest_link::S::size()> v_data;

  // This stores policies
  std::array<weakest_link::A, weakest_link::S::size()> p_data;

  // These are the two tabular functions we need.
  auto V  = rl2::enumerable::make_tabular<weakest_link::S>(v_data.begin());
  auto pi = rl2::enumerable::make_tabular<weakest_link::S>(p_data.begin());

  // We initialize data arrays
  auto random_action_generator = rl2::enumerable::uniform_sampler<weakest_link::A>(gen);
  for(auto& v : v_data) v = 0;
  for(auto& a : p_data) a = random_action_generator();

  // We set parameters
  double epsilon                     = 1e-3; // The threshold for considering variations of V as tiny.
  double gamma                       = .9;   // The gamma parameter of the MDP.
  double correct_answer_probability  = .75;  // The ability of the player to provide good answers.

  // This is the model (T = transition probabilities, R = rewards)
  auto  T = weakest_link::make_Tdistrib(correct_answer_probability);
  auto& R = weakest_link::R;

  
  {
    //
    // Value iteration
    //
    
    // This sets V to V*.
    rl2::critic::dp::enumerable::value_iteration<weakest_link::S, weakest_link::A>(V, T, R, gamma, epsilon);
  }

  {
  }


  return 0;
}
