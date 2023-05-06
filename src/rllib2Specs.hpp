#pragma once
#include <concepts>
#include <cstddef>

namespace rl2 {
  namespace specs {

    /**
     * @short transition function
     */
    template<typename TRANSITION, typename STATE, typename ACTION>
    concept transition =
      requires (TRANSITION const cT, STATE s, STATE const cs, ACTION const ca) {
      s = cT(cs, ca);
    };
      
    /**
     * @short reward function
     */
    template<typename REWARD, typename STATE, typename ACTION>
    concept reward =
      requires (REWARD const cR, STATE s, STATE const cs, ACTION const ca) {
      {cR(cs, ca, cs)} -> std::convertible_to<double>;
    };
      
    /**
     * @short terminal function
     */
    template<typename TERMINAL, typename STATE>
    concept terminal =
      requires (TERMINAL const cT, STATE const cs) {
      {cT(cs)} -> std::convertible_to<bool>;
    };

    /**
     * @short This is for type whose values can be indexed by a positive integer.
     */
    template<typename INDEXED>
    concept indexed =
      requires (std::size_t idx) {
      {idx} -> std::convertible_to<INDEXED>;
    };

    /** 
     * @short This provides a index conversion functions.
     */
    template<typename CONVERTOR, typename BASE>
    concept static_index_conversion =
      requires (BASE const cbase, std::size_t size){
      {CONVERTOR::from(cbase)} -> std::convertible_to<std::size_t>;
      {CONVERTOR::to(size)} -> std::convertible_to<BASE>;
    };
    
  }
}
