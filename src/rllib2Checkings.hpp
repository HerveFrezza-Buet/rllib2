#pragma once

#include <cstddef>

#include <gdyn.hpp>

#include <rllib2Concepts.hpp>
#include <rllib2Enumerable.hpp>
#include <rllib2MDP.hpp>
#include <rllib2Functional.hpp>

namespace rl2 {
  namespace checkings {
    
    // Index conversion
    // ----------------
    
    struct index_conversion {
      static std::size_t from(char base)          {return static_cast<std::size_t>(base) - 65;}
      static char        to  (std::size_t index)  {return static_cast<char>(index + 65);      }
    };
    static_assert(concepts::static_index_conversion<index_conversion, char>);

    // Enumerable
    // ----------
    
    using enumerable_int  = enumerable::count<int,  10>;
    using enumerable_char = enumerable::count<char, 10>;
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
  }
}
