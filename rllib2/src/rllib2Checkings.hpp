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
    static_assert(concepts::enumerable::static_index_conversion<index_conversion, char>);

    // Enumerable
    // ----------
    
    using enumerable_int  = enumerable::set<int,  10>;
    using enumerable_char = enumerable::set<char, 10>;
    static_assert(concepts::enumerable::finite<enumerable_int>);

    // MDP
    // ---
    
    using process = MDP<char, int>;
    static_assert(gdyn::concepts::system<process>);
    static_assert(gdyn::concepts::transparent_system<process>);
    static_assert(concepts::mdp<process>);

    // Two_Args_Function
    // -----------------

    using params_it_type = double*;
    using critic = enumerable::two_args_tabular<enumerable_char, enumerable_int, params_it_type>;
    static_assert(concepts::two_args_function<critic>);

    // Linear approximation
    // --------------------

    using q_state = double;
    using q_action = enumerable_int;
    using s_feature = features::polynomial<8>;
    static_assert(concepts::feature<s_feature, double>);
    
    using theta_params = nuplet::from<double, s_feature::dim * q_action::size()>; 
    static_assert(concepts::nuplet<theta_params>);

    using q_parametrized = linear::enumerable::action::q<theta_params, q_state, q_action, s_feature>;
    static_assert(concepts::enumerable::action::linear_qfunction<q_parametrized>);
    static_assert(concepts::enumerable::action::two_args_function<q_parametrized>);

    using iterable = std::array<double, 10>;
    using nuplet_wrapper = nuplet::by_default::wrapper<nuplet::from_range<iterable, 10>, iterable>;
    static_assert(concepts::nuplet_wrapper<nuplet_wrapper, iterable>);
    
    // TODO : write bellman operator concept checking
  }
}
