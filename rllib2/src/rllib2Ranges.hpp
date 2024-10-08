
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

#include <ranges>

#include <rllib2Concepts.hpp>
#include <rllib2Iterators.hpp>

namespace rl2 {
  namespace ranges {
    
    // #########
    // #       #
    // # SARSA #
    // #       #
    // #########

    template<std::ranges::input_range R>
    requires std::ranges::view<R> &&
    concepts::mdp_orbit_iterator<std::ranges::iterator_t<R>>
    class sarsa_view : public std::ranges::view_interface<sarsa_view<R>> {
    private:
      R from {};
      std::ranges::iterator_t<R> it {std::begin(from)};

    public:

      sarsa_view() = default;
      sarsa_view(R from) : from(from), it(std::begin(from)) {}
      constexpr R base() const & {return from;}
      constexpr R base() &&      {return std::move(from);}
      constexpr auto begin() const {
	return iterators::sarsa<std::ranges::iterator_t<R>,
				std::ranges::sentinel_t<R>>(from.begin(), from.end());
      }
      constexpr auto end()   const {return gdyn::iterators::terminal;}
    };
    
    template<typename R> sarsa_view(R&&) -> sarsa_view<std::ranges::views::all_t<R>>;

    namespace details {
      struct sarsa_range_adaptor_closure {
	constexpr sarsa_range_adaptor_closure() {}
	template <std::ranges::viewable_range R> constexpr auto operator()(R&& from) const {return sarsa_view(std::forward<R>(from));}
      };
      
      struct sarsa_range_adaptor {
	template<std::ranges::viewable_range R> constexpr auto operator()(R&& from) {return sarsa_view(std::forward<R>(from));}
	constexpr auto operator()() {return sarsa_range_adaptor_closure();}
      };
      template <std::ranges::viewable_range R>
      constexpr auto operator | (R&& from, sarsa_range_adaptor_closure const& closure) {return closure(std::forward<R>(from));}
    }
    
    namespace views {
      constexpr auto sarsa = details::sarsa_range_adaptor()();
    }
  
  }
  
  
  
  namespace views = ranges::views;
}
