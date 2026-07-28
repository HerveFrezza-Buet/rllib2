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

#include <utility>
#include <functional>
#include <iterator>
#include <random>
#include <array>
#include <ranges>

#include<rllib2Concepts.hpp>

namespace rl2 {

  template<typename STATE, typename ACTION>
  struct system {
  public:
    using state_type       = STATE;
    using observation_type = STATE; 
    using command_type     = ACTION;
    using report_type      = double;
    
    system()                      = delete;
    system(const system&)            = default;
    system(system&&)                 = default;
    system& operator=(const system&) = default;
    system& operator=(system&&)      = default;
    
  private:
    
    std::function<STATE  (const STATE&, const ACTION&)>               transition_func;
    std::function<double (const STATE&, const ACTION&, const STATE&)> reward_func;
    std::function<bool   (const STATE&)>                              terminal;
    
    state_type current_state;

  public:


    template<concepts::transition_func<STATE, ACTION> TRANSITION_FUNC, concepts::reward_func<STATE, ACTION> REWARD_FUNC, concepts::terminal<STATE> TERMINAL>
    system(const TRANSITION_FUNC& T, const REWARD_FUNC& R, const TERMINAL& terminal)
      : transition_func(T), reward_func(R), terminal(terminal),
	current_state() {}

    void operator=(const state_type& s) {current_state = s;}
    observation_type operator*() const  {return current_state;}
    operator bool() const               {return !terminal(current_state);}
    
    report_type operator()(command_type command) {
      if(*this) {
	auto next_state = transition_func(current_state, command);
	double rew = reward_func(current_state, command, next_state);
	current_state = next_state;
	return rew;
      }
      return 0.; // Transition from a terminal state gives a 0 reward.
    }

    state_type state() const {return current_state;}
  };

  template<typename STATE, typename ACTION, concepts::transition_func<STATE, ACTION> TRANSITION_FUNC, concepts::reward_func<STATE, ACTION> REWARD_FUNC, concepts::terminal<STATE> TERMINAL>
  auto make_system(const TRANSITION_FUNC& T, const REWARD_FUNC& R, const TERMINAL& terminal) {
    return system<STATE, ACTION>(T, R, terminal);
  }

  namespace enumerable {
    template<concepts::enumerable::finite STATE, concepts::enumerable::finite ACTION,
	     typename RANDOM_GENERATOR,
	     concepts::transition_distrib<STATE, ACTION> TRANSITION_DISTRIB>
    auto make_transition_function(RANDOM_GENERATOR& gen, const TRANSITION_DISTRIB& T) {
      std::array<std::array<std::array<double, STATE::size()>, ACTION::size()>, STATE::size()> p;
      std::array<std::array<std::discrete_distribution<std::size_t>, ACTION::size()>, STATE::size()> distribs;
      for(auto s = STATE::begin(); s != STATE::end(); ++s)
	for(auto a = ACTION::begin(); a != ACTION::end(); ++a) {
	  std::size_t s_index = *s;
	  std::size_t a_index = *a;
	  auto& probas = p[s_index][a_index];
	  for(auto& [ss_index, ps] : probas | std::views::enumerate)
	    ps = T(s_index, a_index, static_cast<std::size_t>(ss_index));
	  distribs[s_index][a_index] = std::discrete_distribution<std::size_t>(probas.begin(), probas.end());
	}
      return [&gen, distribs](const STATE& s, const ACTION& a) -> STATE {
	return distribs[static_cast<std::size_t>(s)][static_cast<std::size_t>(a)](gen);
      };
    }
  }
  
}
