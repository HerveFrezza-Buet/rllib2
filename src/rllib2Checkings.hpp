#pragma once

#include <cstddef>

#include <gdyn.hpp>

#include <rllib2Specs.hpp>
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
    static_assert(specs::static_index_conversion<index_conversion, char>);

    // Enumerable
    // ----------
    
    using enumerable_int  = enumerable::count<int,  10>;
    using enumerable_char = enumerable::count<char, 10>;
    static_assert(specs::enumerable<enumerable_int>);

    // MDP
    // ---
    
    using process = MDP<char, int>;
    static_assert(gdyn::specs::system<process>);
    static_assert(specs::mdp<process>);

    // Table
    // -----

    using params_it_type = double*;
    using critic = tabular::table<enumerable_char, enumerable_int, params_it_type>;
    static_assert(specs::table<critic>);
  }
}
