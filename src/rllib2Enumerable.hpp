#pragma once

#include <cstddef>
#include <iterator>
#include <random>
#include <utility>

#include <rllib2Specs.hpp>

namespace rl2 {


  namespace by_default {
    template<specs::indexed BASE>
    struct IndexConversion {
      static std::size_t from(const BASE& base) {return static_cast<std::size_t>(base);}
      static BASE        to(std::size_t index)  {return static_cast<BASE>(index);      }
    };
  }

  
  namespace enumerable {
    template<typename BASE, std::size_t NB,
	     specs::static_index_conversion<BASE> INDEX_CONVERSION = by_default::IndexConversion<BASE>>
    struct count {
      using base_type = BASE;
      struct iterator {
	using value_type = base_type;
	using difference_type = std::ptrdiff_t;
	std::size_t index;
	constexpr iterator()                              = default;
	constexpr iterator(const iterator&)               = default;
	constexpr iterator& operator=(const iterator&)    = default;
	constexpr iterator(std::size_t index) : index(index) {}
	constexpr base_type operator*() const {return INDEX_CONVERSION::to(index);}
	constexpr bool operator<=>(const iterator&) const = default;
	constexpr operator std::size_t () const {return index;}

	constexpr iterator& operator++() {++index; return *this;}
	constexpr iterator  operator++(int) {auto tmp = *this; ++(*this); return tmp;}
	/* to be done : implement operator for random access iterator */
      };

    private:
      base_type value;
      static base_type quantize(const base_type& value) {
	return INDEX_CONVERSION::to(INDEX_CONVERSION::from(value));
      }
      
    public:
      count()              = default;
      count(const count&)  = default;
      count(count&&)       = default;
    
      bool operator<=>(const count&) const = default;
    
      count& operator=(const count&) = default;
      count& operator=(count&&)      = default;

      count(const base_type& value) : value(quantize(value)) {}
      count& operator=(const base_type& value) {this->value = quantize(value); return *this;}
    
      count(std::size_t index) : value(INDEX_CONVERSION::to(index)) {}
      count& operator=(std::size_t index) {this->value = INDEX_CONVERSION::to(index); return *this;}

      count(iterator it) : count(static_cast<std::size_t>(it)) {}
      count& operator=(iterator it) {return (*this = static_cast<std::size_t>(it));}
    
     
      operator std::size_t () const {return INDEX_CONVERSION::from(this->value);}
      operator base_type   () const {return value;}
    
      constexpr static iterator begin   = iterator(0);
      constexpr static iterator end     = iterator(NB);
      constexpr static std::size_t size = NB;
    };
  
    template <specs::enumerable FIRST, specs::enumerable SECOND>
    struct pair_index_conversion {
      static std::size_t from(const std::pair<typename FIRST::base_type, typename SECOND::base_type>& base) {
	return static_cast<std::size_t>(FIRST(base.first)) * SECOND::size + static_cast<std::size_t>(SECOND(base.second));
      }
      static std::pair<typename FIRST::base_type, typename SECOND::base_type>  to(std::size_t index)  {
	return {
	  static_cast<typename FIRST::base_type >(FIRST (index / SECOND::size)),
	  static_cast<typename SECOND::base_type>(SECOND(index % SECOND::size))
	};
      }
    };
    
    template<specs::enumerable FIRST, specs::enumerable SECOND>
    struct pair : public count<std::pair<typename FIRST::base_type, typename SECOND::base_type>, FIRST::size * SECOND::size, pair_index_conversion<FIRST, SECOND>> {
      using super_type = count<std::pair<typename FIRST::base_type, typename SECOND::base_type>, FIRST::size * SECOND::size, pair_index_conversion<FIRST, SECOND>>;

      using super_type::count;
      using super_type::operator=;

      pair(const FIRST& first, const SECOND& second)
	: super_type(typename super_type::base_type(static_cast<typename FIRST::base_type>(first),
						    static_cast<typename SECOND::base_type>(second))) {}
    };

    template<specs::enumerable STATE, specs::enumerable OBSERVATION, specs::enumerable COMMAND,
      gdyn::specs::system SYSTEM>
    requires
    std::same_as<typename STATE::base_type, typename SYSTEM::state_type>
    && std::same_as<typename OBSERVATION::base_type, typename SYSTEM::observation_type>
    && std::same_as<typename COMMAND::base_type, typename SYSTEM::command_type>
    struct system {
      using state_type = STATE;
      using observation_type = OBSERVATION;
      using command_type = COMMAND;
      using report_type = typename SYSTEM::report_type;
      
      SYSTEM& borrowed_system;
      system(SYSTEM& borrowed_system) : borrowed_system(borrowed_system) {}
      system() = delete;
      system(system&&) = delete;
      system& operator=(system&&) = delete;
      
      void operator=(const state_type& state)             {borrowed_system = static_cast<STATE::base_type>(state);}
      report_type operator()(const command_type& command) {return borrowed_system(static_cast<COMMAND::base_type>(command));}
      observation_type operator*() const                  {return *borrowed_system;}
      operator bool() const                               {return borrowed_system;}
    };

    namespace utils {
      namespace digitize {
	

	// Partition in bins, upper bound in last bin
	std::size_t to_index(double value, double value_min, double value_max, std::size_t nb_bins)
	{
	  if (value == value_max) return nb_bins-1;
	  double reloc = (value - value_min) / (value_max - value_min) * static_cast<double>(nb_bins);
	  return static_cast<std::size_t>(reloc);
	}

	// When converting from bin, give the middle value of the bin
	double to_value(std::size_t index, double value_min, double value_max, std::size_t nb_bins)
	{
	  return value_min + (static_cast<double>(index) + 0.5)
	    * (value_max - value_min) / static_cast<double>(nb_bins);
	}
	
	std::size_t to_index(double value, std::size_t nb_bins) {
	  return to_index(value, 0., 1., nb_bins);
	}
	
	double to_value(std::size_t index, std::size_t nb_bins) {
	  return to_value(index, 0., 1., nb_bins);
	}

      }
    }
  }
}
