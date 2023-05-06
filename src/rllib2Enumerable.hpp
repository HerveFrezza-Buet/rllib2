#pragma once

#include <cstddef>
#include <iterator>

#include <rllib2Specs.hpp>

namespace rl2 {


  namespace by_default {
    template<specs::indexed BASE>
    struct IndexConversion {
      static std::size_t from(const BASE& base) {return static_cast<std::size_t>(base);}
      static BASE        to(std::size_t index)  {return static_cast<BASE>(index);      }
    };
  }
  
  template<typename BASE, std::size_t NB, specs::static_index_conversion<BASE> INDEX_CONVERSION>
  struct enumerable_common {
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
    enumerable_common()                                    = default;
    enumerable_common(const enumerable_common&)            = default;
    enumerable_common(enumerable_common&&)                 = default;
    enumerable_common& operator=(const enumerable_common&) = default;
    enumerable_common& operator=(enumerable_common&&)      = default;

    enumerable_common(const base_type& value) : value(value) {}
    enumerable_common& operator=(const base_type& value) {this->value = value;}
     
    operator base_type () const {return value;}
    
    constexpr static iterator begin   = iterator(0);
    constexpr static iterator end     = iterator(NB);
    constexpr static std::size_t size = NB;
  };

  
  template<typename BASE, std::size_t NB, typename INDEX_CONVERSION = by_default::IndexConversion<BASE>>
  struct enumerable: public enumerable_common<BASE, NB, INDEX_CONVERSION> {
    using enumerable_common<BASE, NB, INDEX_CONVERSION>::enumerable_common;
  };
  
  
}
