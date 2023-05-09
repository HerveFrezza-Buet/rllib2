#pragma once
#include <concepts>
#include <cstddef>
#include <tuple>

namespace rl2 {
  namespace specs {

    /**
     * @short transition function
     */
    template<typename TRANSITION, typename STATE, typename ACTION>
    concept transition =
      requires (TRANSITION const cT, STATE const cs, ACTION const ca) {
      {cT(cs, ca)} -> std::same_as<STATE>;
    };
      
    /**
     * @short reward function
     */
    template<typename REWARD, typename STATE, typename ACTION>
    concept reward =
      requires (REWARD const cR, STATE s, STATE const cs, ACTION const ca) {
      {cR(cs, ca, cs)} -> std::same_as<double>;
    };
      
    /**
     * @short terminal function
     */
    template<typename TERMINAL, typename STATE>
    concept terminal =
      requires (TERMINAL const cT, STATE const cs) {
      {cT(cs)} -> std::same_as<bool>;
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
      {CONVERTOR::from(cbase)} -> std::same_as<std::size_t>;
      {CONVERTOR::to(size)} -> std::same_as<BASE>;
    };

    /**
     * @short Sets whose values can be enumerated.
     */
    template<typename ENUMERABLE>
    concept enumerable =
      requires {
      typename ENUMERABLE::base_type;
      typename ENUMERABLE::iterator;
      // {ENUMERABLE::begin} -> std::same_as<typename ENUMERABLE::iterator>;
      // {ENUMERABLE::end}   -> std::same_as<typename ENUMERABLE::iterator>;
      // {ENUMERABLE::size}  -> std::same_as<std::size_t>;
      {ENUMERABLE::begin} -> std::convertible_to<typename ENUMERABLE::iterator>;
      {ENUMERABLE::end}   -> std::convertible_to<typename ENUMERABLE::iterator>;
      {ENUMERABLE::size}  -> std::convertible_to<std::size_t>;
    } &&
    requires (ENUMERABLE::iterator it) {
      ++it;
      {*it} -> std::same_as<typename ENUMERABLE::base_type>;
    };

    /**
     * @short A MDP system
     */
    template<typename MDP>
    concept mdp =
      gdyn::specs::system<MDP>
      && std::same_as<typename MDP::observation_type, std::pair<typename MDP::state_type, double>>;

    /**
     * @short orbit point when the system is a MDP 
     */
    template<typename ORBIT_VALUE>
    concept mdp_orbit_point =
      gdyn::specs::orbit_point<ORBIT_VALUE>
      && std::same_as<typename ORBIT_VALUE::observation_type::second_type, double>;

    /**
     * @short orbit iterator when the system is a MDP
     */
    template<typename ORBIT_ITERATOR>
    concept mdp_orbit_iterator =
      gdyn::specs::orbit_iterator<ORBIT_ITERATOR>
      && mdp_orbit_point<std::iter_value_t<ORBIT_ITERATOR>>;

    /**
     * @short Specifies a producer (i.e. res = f()) whose result type is enumerable
     */
    template<typename F>
    concept enumerable_producer =
      std::invocable<F>
      && enumerable<std::invoke_result_t<F, void>>;
  }
}
