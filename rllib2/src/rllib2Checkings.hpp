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
    

    using q_state = double;
    using q_action = enumerable_int;
    using s_feature = features::polynomial<8>;
    static_assert(concepts::feature<s_feature, double>);
    
    using theta_params = nuplet::from<double, s_feature::dim * q_action::size()>; 
    static_assert(concepts::nuplet<theta_params>);

    using q_parametrized = linear::discrete_a::q<theta_params, q_state, q_action, s_feature>;
    static_assert(concepts::discrete_a::linear_qfunction<q_parametrized>);
    static_assert(concepts::discrete_a::two_args_function<q_parametrized>);
    
    
    // To do : write bellman operator concept checking
  }
}
