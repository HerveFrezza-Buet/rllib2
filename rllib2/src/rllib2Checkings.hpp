#pragma once

#include <cstddef>

#include <gdyn.hpp>

#include <rllib2Concepts.hpp>
#include <rllib2Enumerable.hpp>
#include <rllib2MDP.hpp>
#include <rllib2Functional.hpp>
#include <rllib2Features.hpp>
#include <rllib2Nuplet.hpp>
#include <rllib2Transition.hpp>

namespace rl2 {
  namespace checkings {

    // Transitions
    // -----------
    static_assert(concepts::sarsa<sarsa<char, int>, char, int>);
    
    // Index conversion
    // ----------------
    
    struct index_conversion {
      static std::size_t from(char base)          {return static_cast<std::size_t>(base) - 65;}
      static char        to  (std::size_t index)  {return static_cast<char>(index + 65);      }
    };
    static_assert(concepts::static_index_conversion<index_conversion, char>);

    // Enumerable
    // ----------
    
    using enumerable_int  = enumerable::set<int,  10>;
    using enumerable_char = enumerable::set<char, 10>;
    static_assert(concepts::enumerable<enumerable_int>);

    // MDP
    // ---
    
    using process = MDP<char, int>;
    static_assert(gdyn::concepts::system<process>);
    static_assert(gdyn::concepts::transparent_system<process>);
    static_assert(concepts::mdp<process>);

    // Two_Args_Function
    // -----------------

    using params_it_type = double*;
    using critic = tabular::two_args_function<enumerable_char, enumerable_int, params_it_type>;
    static_assert(concepts::two_args_function<critic>);

    // Linear approximation
    // --------------------
    
    using theta_params = nuplet::from<std::vector<double>, 32>; 
    static_assert(concepts::nuplet<theta_params>);

    using polynomial_feature = features::polynomial<8>;
    static_assert(concepts::feature<polynomial_feature, double>);

    // using q_parametrized = linear::discrete_a::q<theta_params, double, enumerable_int, state_feature>;
    
    
    // To do : write bellman operator concept checking
  }
}
