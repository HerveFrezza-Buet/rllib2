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

    
    // sarsa
    
    template<concepts::orbit_iterator ORBIT_ITERATOR,
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




    // This is for iterating on system borrowed_orbits.
    
    template<concepts::enumerable::action::wrapped_system SYSTEM,
	     gdyn::concepts::command_iterator<typename SYSTEM::command_type> COMMAND_ITERATOR,
	     typename COMMAND_SENTINEL>
    struct borrowed_orbit {
      
    private:
      SYSTEM* system = nullptr;
      COMMAND_ITERATOR it;
      COMMAND_SENTINEL end;

    public:

      using difference_type = std::ptrdiff_t;

      
      struct value_type {
	using observation_type = typename SYSTEM::borrowed_system_type::observation_type;
	using command_type     = typename SYSTEM::borrowed_system_type::command_type;
	using report_type      = typename SYSTEM::report_type;
	observation_type            current_observation;
	std::optional<command_type> next_command;
	std::optional<report_type>  previous_report;
	
	
	value_type()                             = default;
	value_type(const value_type&)            = default;
	value_type& operator=(const value_type&) = default;
	value_type(value_type&&)                 = default;
	value_type& operator=(value_type&&)      = default;
      };
      
    private:
      
      value_type value;
      bool terminated = false;
      
    public:
      
      
      borrowed_orbit()                        = delete;
      borrowed_orbit(const borrowed_orbit&)            = default;
      borrowed_orbit(borrowed_orbit&&)                 = default;
      borrowed_orbit& operator=(const borrowed_orbit&) = default;
      borrowed_orbit& operator=(borrowed_orbit&&     ) = default;

      borrowed_orbit(SYSTEM& system, COMMAND_ITERATOR it, COMMAND_SENTINEL end)
	: system(&system), it(it), end(end),
	  value(), terminated() {
	  if(it == end)
	    terminated = true;
	  else if(system) {
	    value.current_observation = *(system.borrowed_system);
	    value.next_command = static_cast<typename SYSTEM::command_type::base_type>(*it);
	  }
	  else { // We are in a terminal state.
	    value.current_observation = *(system.borrowed_system);
	    value.next_command = std::nullopt;
	  }
      }
      
      bool operator==(gdyn::iterators::terminal_t) const {return terminated;}
      auto& operator*() const {return value;}
      auto& operator++() {
	if(value.next_command) {// we are not in a terminal state (the has been checked at previous iteration).
	  // We perform a transition.
	  value.previous_report = (system->borrowed_system)(*(value.next_command));
	  value.current_observation = *(system->borrowed_system);
	  
	  ++it; // We get next command
	  if(it == end || !(*system))
	    value.next_command = std::nullopt;
	  else
	    value.next_command = static_cast<typename SYSTEM::command_type::base_type>(*it);
	}
	else // we are in a terminal state
	  terminated = true;
	return *this;
      }
      auto  operator++(int) {auto res = *this; ++(*this); return res;}   
    };



  }
}
