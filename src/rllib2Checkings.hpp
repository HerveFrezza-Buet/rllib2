#pragma once

#include <cstddef>

#include <gdyn.hpp>

#include <rllib2Specs.hpp>
#include <rllib2Enumerable.hpp>
#include <rllib2MDP.hpp>

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
    
    using enumerable_int = enumerable::count<int, 10>;
    static_assert(specs::enumerable<enumerable_int>);

    // MDP
    // ---
    
    using process = MDP<char, int>;
    static_assert(gdyn::specs::system<process>);
    static_assert(specs::mdp<process>);
  }
}
