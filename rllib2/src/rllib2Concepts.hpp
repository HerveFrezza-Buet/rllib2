#pragma once
#include <concepts>
#include <cstddef>
#include <tuple>

namespace rl2 {
  namespace concepts {

    /**
     * @short sarsa-like transitions
     */
    template<typename SARSA, typename STATE, typename ACTION>
    concept sarsa =
      requires(SARSA& ct) {
      {ct.s}              -> std::same_as<STATE&>;
      {ct.a}              -> std::same_as<ACTION&>;
      {ct.r}              -> std::same_as<double&>;
      {ct.ss}             -> std::same_as<STATE&>;
      {ct.aa.has_value()} -> std::convertible_to<bool>;
      {*(ct.aa)}          -> std::same_as<ACTION&>;
      {ct.is_terminal()}  -> std::same_as<bool>;
    };
    
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
      {ENUMERABLE::begin()} -> std::convertible_to<typename ENUMERABLE::iterator>;
      {ENUMERABLE::end()}   -> std::convertible_to<typename ENUMERABLE::iterator>;
      {ENUMERABLE::size()}  -> std::convertible_to<std::size_t>;
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
      gdyn::concepts::system<MDP>
      && std::same_as<typename MDP::report_type, double>;

    /**
     * @short orbit point when the system is a MDP 
     */
    template<typename ORBIT_VALUE>
    concept mdp_orbit_point =
      gdyn::concepts::orbit_point<ORBIT_VALUE>
      && std::same_as<typename ORBIT_VALUE::report_type, double>;

    /**
     * @short orbit iterator when the system is a MDP
     */
    template<typename ORBIT_ITERATOR>
    concept mdp_orbit_iterator =
      gdyn::concepts::orbit_iterator<ORBIT_ITERATOR>
      && mdp_orbit_point<std::iter_value_t<ORBIT_ITERATOR>>;

    /**
     * @short A RL policy
     */
    template<typename POLICY, typename S, typename A>
    concept policy =
      std::invocable<POLICY, S>
      && requires(POLICY const cp, S const cs){
      {cp(cs)} -> std::convertible_to<A>;
    };

    /**
     * @short A function f(a, b), for which f(a) is a function : f(a,b) = f(a)(b).
     */
    template<typename TWO_ARGS_FUNCTION>
    concept two_args_function =
      requires {
      typename TWO_ARGS_FUNCTION::result_type;
      typename TWO_ARGS_FUNCTION::first_entry_type;
      typename TWO_ARGS_FUNCTION::second_entry_type;
    }
      && std::invocable<TWO_ARGS_FUNCTION, typename TWO_ARGS_FUNCTION::first_entry_type, typename TWO_ARGS_FUNCTION::second_entry_type>
    && std::invocable<TWO_ARGS_FUNCTION, typename TWO_ARGS_FUNCTION::first_entry_type>
    && requires(TWO_ARGS_FUNCTION const ct, typename TWO_ARGS_FUNCTION::first_entry_type const cs) {
      typename TWO_ARGS_FUNCTION::first_entry_type;
      typename TWO_ARGS_FUNCTION::second_entry_type;
      {ct(cs)} -> std::invocable<typename TWO_ARGS_FUNCTION::second_entry_type>;
    };

    /**
     * @short A two-args function whose types are enumerable.
     */
    template<typename TWO_ARGS_FUNCTION>
    concept tabular_two_args_function =
      two_args_function<TWO_ARGS_FUNCTION>
      && requires {
      typename TWO_ARGS_FUNCTION::params_iterator_type;
    }
      && enumerable<typename TWO_ARGS_FUNCTION::first_entry_type>
    && enumerable<typename TWO_ARGS_FUNCTION::second_entry_type>;


    /**
     * @short A function such as q(s, a) is scalar.
     */
    template<typename Q, typename S, typename A>
    concept q_function = 
      std::invocable<Q, S, A>
      && requires(Q const cq, S const cs, A const ca) {
      {cq(cs, ca)} -> std::convertible_to<double>;
    };
    
    /**
     * @short A bellman operator
     */
    template<typename OP, typename Q, typename S, typename A, typename TRANS>
    concept bellman_operator =
      sarsa<TRANS, S, A>
      && requires(const OP cop, const Q cq, double gamma, const TRANS ct) {
      {cop(cq, gamma, ct)} -> std::same_as<double>;
    };

    /**
     * @short This is a range whose size is known at compiling time.
     */
    template<typename PARAMS>
    concept nuplet =
      std::ranges::input_range<PARAMS>
      && requires () {
      {PARAMS::dim} -> std::convertible_to<const std::size_t>;
    };

    /**
     * @short This is the feature part (phi in f(x) = theta.T x phi(s)) of a linearly parametrized function.
     */
    template<typename FEATURE, typename X>
    concept feature =
      std::copy_constructible<FEATURE>
      && requires(const FEATURE cf, const X cx) {
      {cf(cx)} -> std::ranges::input_range;
      {cf.dim} -> std::convertible_to<const std::size_t>;
    };
    
  }
}
