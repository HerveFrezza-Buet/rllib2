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

#include <optional>

#include <gdyn.hpp>

#include <rllib2Concepts.hpp>
#include <rllib2Transition.hpp>

namespace rl2 {
  namespace iterators {

    
    // transition
    
    template<concepts::mdp_orbit_iterator ORBIT_ITERATOR,
	     typename ORBIT_SENTINEL>
    struct sarsa {
      
    private:
      ORBIT_ITERATOR it;
      ORBIT_SENTINEL end;

    public:

      using value_type = rl2::sarsa<typename gdyn::iterators::observation_t<ORBIT_ITERATOR>,
				    gdyn::iterators::command_t<ORBIT_ITERATOR>>;
      
    private:
      
      std::optional<value_type> value;
      
      
    public:
      
      using difference_type = std::ptrdiff_t;
      
      sarsa()                             = default;
      sarsa(const sarsa&)                  = default;
      sarsa(sarsa&&)                       = default;
      sarsa& operator=(const sarsa& other) = default;
      sarsa& operator=(sarsa&&      other) = default;

      sarsa(ORBIT_ITERATOR begin, ORBIT_SENTINEL end) : it(begin), end(end), value() {
	if(it != end) {
	  auto start = *(it++);
	  if(it != end) 
	    value = make_sarsa(start, *it);
	}
      }
      
      bool operator==(gdyn::iterators::terminal_t) const {return it == end || !value;}
      
      auto& operator++() {
	++it;
	if(it == end)
	  value = std::nullopt;
	else
	  *value += *it; // We skip to the next sarsa.
	return *this;
      }
      const auto& operator*() const {return *value;} 
      auto  operator++(int)         {auto res = *this; ++(*this); return res;}   
    };


  }
}
