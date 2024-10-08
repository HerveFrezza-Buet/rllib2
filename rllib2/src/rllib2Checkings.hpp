/*

Copyright 2024 Herve FREZZA-BUET, Alain DUTECH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

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
    using transition = sarsa<char, int>;
    static_assert(concepts::sarsa<transition, char, int>);
    
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

    // Q functions
    // -----------
    
    using q_state = double;
    using q_action = enumerable_int;
    using q_transition = sarsa<q_state, q_action>;
    static_assert(concepts::q_function<double (q_state, q_action), q_state, q_action>);
    
    // Linear approximation
    // --------------------

    using s_feature = features::polynomial<8>;
    static_assert(concepts::feature<s_feature, double>);
    
    using theta_params = nuplet::from<double, s_feature::dim * q_action::size()>; 
    static_assert(concepts::nuplet<theta_params>);

    using q_parametrized = linear::enumerable::action::q<theta_params, q_state, q_action, s_feature>;
    static_assert(concepts::q_function<q_parametrized, q_state, q_action>);
    static_assert(concepts::enumerable::action::linear_qfunction<q_parametrized>);
    static_assert(concepts::enumerable::action::two_args_function<q_parametrized>);

    using iterable = std::array<double, 10>;
    using nuplet_wrapper = nuplet::by_default::wrapper<nuplet::from_range<iterable, 10>, iterable>;
    static_assert(concepts::nuplet_wrapper<nuplet_wrapper, iterable>);
    
    // The Bellman operator
    // ---------------------

    double bellman_op(const q_parametrized&, double, const q_transition&);
    using bellman_op_type = decltype(bellman_op);
    static_assert(concepts::bellman_operator<bellman_op_type, q_parametrized, q_state, q_action, q_transition>);
  }
}
