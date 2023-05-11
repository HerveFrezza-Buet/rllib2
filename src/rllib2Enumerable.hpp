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
    
      base_type value;
      count()                   = default;
      count(const count&)  = default;
      count(count&&)       = default;
    
      bool operator<=>(const count&) const = default;
    
      count& operator=(const count&) = default;
      count& operator=(count&&)      = default;

      count(const base_type& value) : value(value) {}
      count& operator=(const base_type& value) {this->value = value; return *this;}
    
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
	return static_cast<std::size_t>(base.first) * SECOND::size + static_cast<std::size_t>(base.second);
      }
      static std::pair<typename FIRST::base_type, typename SECOND::base_type>  to(std::size_t index)  {
	return {
	  static_cast<typename FIRST::base_type >(FIRST (index / SECOND::size)),
	  static_cast<typename SECOND::base_type>(SECOND(index % SECOND::size))
	};
      }
    };
    
    template<specs::enumerable FIRST, specs::enumerable SECOND>
    using pair = count<std::pair<typename FIRST::base_type, typename SECOND::base_type>, FIRST::size * SECOND::size, pair_index_conversion<FIRST, SECOND>>;
  }
}
