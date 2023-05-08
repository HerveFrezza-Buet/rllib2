
#pragma once

#include <ranges>

#include <rllib2Specs.hpp>
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
    specs::mdp_orbit_iterator<std::ranges::iterator_t<R>>
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
