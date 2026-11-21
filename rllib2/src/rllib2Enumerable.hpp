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
#include <iterator>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>

#include <rllib2Concepts.hpp>
#include <rllib2Index.hpp>

namespace rl2 {


  namespace by_default {
    template<rl2::concepts::enumerable::obvious_index BASE>
    struct IndexConversion {
      static rl2::enumerable::index from(const BASE& base)          {return {static_cast<std::size_t>(base)};}
      static BASE                   to(rl2::enumerable::index idx)  {return {idx.value};                     }
    };
  }

  
  namespace enumerable {


    
    template<typename BASE, std::size_t NB,
	     concepts::enumerable::static_index_conversion<BASE> INDEX_CONVERSION = by_default::IndexConversion<BASE>>
    struct set {
      using base_type = BASE;
      struct iterator {
	using value_type = base_type;
	using difference_type = std::ptrdiff_t;
	index idx;
	constexpr iterator()                              = default;
	constexpr iterator(const iterator&)               = default;
	constexpr iterator& operator=(const iterator&)    = default;
	constexpr iterator(index idx) : idx(idx) {}
	constexpr base_type operator*() const {return INDEX_CONVERSION::to(idx);}
	constexpr bool operator<=>(const iterator&) const = default;
	constexpr operator index () const {return idx;}

	constexpr iterator& operator++() {++idx; return *this;}
	constexpr iterator  operator++(int) {auto tmp = *this; ++(*this); return tmp;}
      };

    private:
      base_type value;
      static base_type quantize(const base_type& value) {
	return INDEX_CONVERSION::to(INDEX_CONVERSION::from(value));
      }
      
    public:
      set()            = default;
      set(const set&)  = default;
      set(set&&)       = default;
    
      bool operator<=>(const set&) const = default;
    
      set& operator=(const set&) = default;
      set& operator=(set&&)      = default;

      set(const base_type& value) : value(quantize(value)) {}
      set& operator=(const base_type& value) {this->value = quantize(value); return *this;}
    
      set(index idx) : value(INDEX_CONVERSION::to(idx)) {}
      set& operator=(index idx) {this->value = INDEX_CONVERSION::to(idx); return *this;}

      set(iterator it) : set(static_cast<index>(it)) {}
      set& operator=(iterator it) {return (*this = static_cast<index>(it));}
    
     
      operator index () const {return INDEX_CONVERSION::from(this->value);}
      operator base_type   () const {return value;}
    
      constexpr static iterator begin()   {return iterator(0);}
      constexpr static iterator end()     {return iterator(NB);}
      constexpr static std::size_t size() {return NB;}
    };


    



    
    template <concepts::enumerable::finite FIRST, concepts::enumerable::finite SECOND>
    struct pair_index_conversion {
      static index from(const std::pair<typename FIRST::base_type, typename SECOND::base_type>& base) {
	return static_cast<index>(FIRST(base.first)) * SECOND::size() + static_cast<index>(SECOND(base.second));
      }
      static std::pair<typename FIRST::base_type, typename SECOND::base_type>  to(index idx)  {
	return {
	  static_cast<typename FIRST::base_type >(FIRST (idx / SECOND::size())),
	  static_cast<typename SECOND::base_type>(SECOND(idx % SECOND::size()))
	};
      }
    };
    
    template<concepts::enumerable::finite FIRST, concepts::enumerable::finite SECOND>
    struct pair : public set<std::pair<typename FIRST::base_type, typename SECOND::base_type>, FIRST::size() * SECOND::size(), pair_index_conversion<FIRST, SECOND>> {
      using super_type = set<std::pair<typename FIRST::base_type, typename SECOND::base_type>, FIRST::size() * SECOND::size(), pair_index_conversion<FIRST, SECOND>>;

      using super_type::set;
      using super_type::operator=;

      pair(const FIRST& first, const SECOND& second)
	: super_type(typename super_type::base_type(static_cast<typename FIRST::base_type>(first),
						    static_cast<typename SECOND::base_type>(second))) {}
    };

    template<concepts::enumerable::finite SET>
    std::ostream& operator<<(std::ostream& os, const SET& s) {return os << static_cast<index>(s).value;}
    
    template<concepts::enumerable::finite SET>
    std::istream& operator>>(std::istream& is, SET& s) {
      index idx;
      is >> idx.value;
      s = idx;
      return is;
    }

    template<concepts::enumerable::finite STATE, concepts::enumerable::finite OBSERVATION, concepts::enumerable::finite COMMAND,
      gdyn::concepts::system SYSTEM>
    requires
    std::same_as<typename STATE::base_type, typename SYSTEM::state_type>
    && std::same_as<typename OBSERVATION::base_type, typename SYSTEM::observation_type>
    && std::same_as<typename COMMAND::base_type, typename SYSTEM::command_type>
    struct system {
      using state_type = STATE;
      using observation_type = OBSERVATION;
      using command_type = COMMAND;
      using report_type = typename SYSTEM::report_type;
      using borrowed_system_type = SYSTEM;
      
      SYSTEM& borrowed_system;
      system(SYSTEM& borrowed_system) : borrowed_system(borrowed_system) {}
      system() = delete;
      system(system&&) = delete;
      system& operator=(system&&) = delete;
      
      void operator=(const state_type& state)             {borrowed_system = static_cast<STATE::base_type>(state);}
      report_type operator()(const command_type& command) {return borrowed_system(static_cast<COMMAND::base_type>(command));}
      observation_type operator*() const                  {return *borrowed_system;}
      operator bool() const                               {return borrowed_system;}

      state_type state() const requires(gdyn::concepts::transparent_system<SYSTEM>) {return borrowed_system.state();}
    };


    namespace action {
      template<typename STATE, typename OBSERVATION, concepts::enumerable::finite COMMAND,
	       gdyn::concepts::system SYSTEM>
      requires std::same_as<typename COMMAND::base_type, typename SYSTEM::command_type>
      struct system {
	using state_type = STATE;
	using observation_type = OBSERVATION;
	using command_type = COMMAND;
	using report_type = typename SYSTEM::report_type;
	using borrowed_system_type = SYSTEM;
      
	SYSTEM& borrowed_system;
	system(SYSTEM& borrowed_system) : borrowed_system(borrowed_system) {}
	system() = delete;
	system(system&&) = delete;
	system& operator=(system&&) = delete;
      
	void operator=(const state_type& state)             {borrowed_system = state;}
	report_type operator()(const command_type& command) {return borrowed_system(static_cast<COMMAND::base_type>(command));}
	observation_type operator*() const                  {return *borrowed_system;}
	operator bool() const                               {return borrowed_system;}

	state_type state() const requires(gdyn::concepts::transparent_system<SYSTEM>) {return borrowed_system.state();}
      };
    }

    namespace utils {
      namespace digitize {

	
	template<bool AT_MIN, bool AT_MAX>
	double nb_bins_in_range(std::size_t nb_bins) {
	  if(AT_MIN and AT_MAX) return nb_bins - 1;
	  if(AT_MIN  or AT_MAX) return nb_bins - .5;
	  return nb_bins;
	}

	template<bool AT_MIN=false, bool AT_MAX=false>
	index to_index(double value, double value_min, double value_max, std::size_t nb_bins)
	{
	  double reloc = (value - value_min) / (value_max - value_min) * nb_bins_in_range<AT_MIN, AT_MAX>(nb_bins);
	  if(AT_MIN) reloc += .5;
	  return {static_cast<std::size_t>(std::clamp(reloc, static_cast<double>(0),  static_cast<double>(nb_bins - 1)))};
	}

	template<bool AT_MIN=false, bool AT_MAX=false>
	double to_value(index idx, double value_min, double value_max, std::size_t nb_bins)
	{
	  double size {static_cast<double>(idx.value)};
	  if(not AT_MIN) size += .5;
	  return value_min + size * (value_max - value_min) / nb_bins_in_range<AT_MIN, AT_MAX>(nb_bins);
	}
	
	template<bool AT_MIN=false, bool AT_MAX=false>
	index to_index(double value, std::size_t nb_bins) {
	  return to_index<AT_MIN, AT_MAX>(value, 0., 1., nb_bins);
	}
	
	template<bool AT_MIN=false, bool AT_MAX=false>
	double to_value(index idx, std::size_t nb_bins) {
	  return to_value<AT_MIN, AT_MAX>(idx, 0., 1., nb_bins);
	}

	/**
	 * This builds a enumerable set for digitized scalar.
	 */
	template<std::size_t NB, double MIN, double MAX, bool AT_MIN=false, bool AT_MAX=false>
	struct scalar {
	public:
	  struct convertor {
	    static constexpr size_t size() {return NB;}; 
	    static double to(index idx) {
	      return to_value<AT_MIN, AT_MAX>(idx, MIN, MAX, NB);
	    }
	    static index from(double value) {
	      if(value < MIN) return 0;
	      if(value >= MAX) return NB-1;
	      return to_index<AT_MIN, AT_MAX>(value, MIN, MAX, NB);
	    }
	  };
	public:
	  using set = rl2::enumerable::set<double, NB, convertor>; 
	};
      }
    }
  }
}
