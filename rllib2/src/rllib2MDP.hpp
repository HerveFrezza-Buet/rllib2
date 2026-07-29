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
#include <vector>
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
    namespace details {
      template<concepts::enumerable::finite STATE,
	       concepts::enumerable::finite ACTION, 
	       typename GEN>
      struct discrete_distributions {
	using distribs_type = std::vector<std::discrete_distribution<std::size_t>>;
	mutable distribs_type distribs;
	GEN& gen;
	discrete_distributions(distribs_type&& distribs,
			       GEN& gen)
	  : distribs(std::forward<distribs_type>(distribs)), gen(gen) {}
	
	STATE operator()(const STATE& s, const ACTION& a) const {
	  return distribs[static_cast<std::size_t>(s) * ACTION::size() + static_cast<std::size_t>(a)](gen);
	}
      };
    }


    
    template<concepts::enumerable::finite STATE,
	     concepts::enumerable::finite ACTION,
	     typename GEN,
	     concepts::transition_distrib<STATE, ACTION> TRANSITION_DISTRIB>
    auto make_transition_function(GEN& gen, const TRANSITION_DISTRIB& T) {
      typename details::discrete_distributions<STATE, ACTION, GEN>::distribs_type distribs;
      distribs.reserve(STATE::size() * ACTION::size());
      std::array<double, STATE::size()> p;
      for(auto s_it = STATE::begin(); s_it != STATE::end(); ++s_it)
	for(auto a_it = ACTION::begin(); a_it != ACTION::end(); ++a_it) {
	  auto p_it = p.begin();
	  for(auto ss_it = STATE::begin(); ss_it != STATE::end(); ++ss_it)
	    *p_it++ = T(*s_it, *a_it, *ss_it);
	  distribs.emplace_back(p.begin(), p.end());
	}
      return details::discrete_distributions<STATE, ACTION, GEN>(std::move(distribs), gen);
    }
  }
  
}

