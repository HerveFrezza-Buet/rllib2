#pragma once

#include <cstddef>
#include <iterator>
#include <random>

#include <rllib2Specs.hpp>

namespace rl2 {


  namespace by_default {
    template<specs::indexed BASE>
    struct IndexConversion {
      static std::size_t from(const BASE& base) {return static_cast<std::size_t>(base);}
      static BASE        to(std::size_t index)  {return static_cast<BASE>(index);      }
    };
  }
  
  template<typename BASE, std::size_t NB,
	   specs::static_index_conversion<BASE> INDEX_CONVERSION = by_default::IndexConversion<BASE>>
  struct enumerable {
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
      constexpr iterator  operator++(int) {auto tmp = *this; (*this)++; return tmp;}
      /* to be done : implement operator for random access iterator */
    };
    
    base_type value;
    enumerable()                   = default;
    enumerable(const enumerable&)  = default;
    enumerable(enumerable&&)       = default;
    
    bool operator<=>(const enumerable&) const = default;
    
    enumerable& operator=(const enumerable&) = default;
    enumerable& operator=(enumerable&&)      = default;

    enumerable(const base_type& value) : value(value) {}
    enumerable& operator=(const base_type& value) {this->value = value; return *this;}
    
    enumerable(std::size_t index) : value(INDEX_CONVERSION::to(index)) {}
    enumerable& operator=(std::size_t index) {this->value = INDEX_CONVERSION::to(index); return *this;}

    enumerable(iterator it) : enumerable(static_cast<std::size_t>(it)) {}
    enumerable& operator=(iterator it) {return (*this = static_cast<std::size_t>(it));}
    
     
    operator std::size_t () const {return INDEX_CONVERSION::from(this->value);}
    operator base_type   () const {return value;}
    
    constexpr static iterator begin   = iterator(0);
    constexpr static iterator end     = iterator(NB);
    constexpr static std::size_t size = NB;
  };

  
}
