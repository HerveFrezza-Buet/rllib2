#pragma once
#include <concepts>
#include <cstddef>
#include <tuple>
#include <array>
#include <type_traits>
#include <memory>

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

    namespace enumerable {
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
      template<typename FINITE>
      concept finite =
      requires {
	typename FINITE::base_type;
	typename FINITE::iterator;
	{FINITE::begin()} -> std::convertible_to<typename FINITE::iterator>;
	{FINITE::end()}   -> std::convertible_to<typename FINITE::iterator>;
	{FINITE::size()}  -> std::convertible_to<std::size_t>;
      } &&
      requires (FINITE::iterator it) {
	++it;
	{*it} -> std::same_as<typename FINITE::base_type>;
      };
    }

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
    concept policy = gdyn::concepts::controller<POLICY, S, A>;

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

    namespace enumerable {
      /**
       * @short A two-args function whose types are finite.
       */
      template<typename TWO_ARGS_FUNCTION>
      concept two_args_function =
      rl2::concepts::two_args_function<TWO_ARGS_FUNCTION>
	&& requires {
	typename TWO_ARGS_FUNCTION::params_iterator_type;
      }
	&& finite<typename TWO_ARGS_FUNCTION::first_entry_type>
      && finite<typename TWO_ARGS_FUNCTION::second_entry_type>;

      namespace action {
	/**
	 * @short A two-args function whose second type is finite.
	 */
	template<typename TWO_ARGS_FUNCTION>
	concept two_args_function =
	concepts::two_args_function<TWO_ARGS_FUNCTION>
	  && finite<typename TWO_ARGS_FUNCTION::second_entry_type>;
      }
    }


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
    template<typename X>
    concept nuplet =
    std::ranges::input_range<X>
      && requires () {
      {X::dim} -> std::convertible_to<const std::size_t>;
    };

    /**
     * @short This is the feature part (phi in f(x) = theta.T x phi(s)) of a linearly parametrized function.
     */
    template<typename FEATURE, typename X>
    concept feature =
    std::copy_constructible<FEATURE>
      && requires(const FEATURE cf, const X cx) {
      {cf(cx)} -> std::ranges::input_range;
      {FEATURE::dim} -> std::convertible_to<const std::size_t>;
    };

    template<typename WRAPPER, typename X>
    concept nuplet_wrapper =
    std::constructible_from<WRAPPER, X>
      && requires (const WRAPPER cw, const X cx) {
      {cw.begin()} -> std::input_iterator;
      {cw.end()} -> std::sentinel_for<decltype(cw.begin())>;
      {WRAPPER::dim} -> std::convertible_to<const std::size_t>;
    };
    

    // /**
    //  * @short A concept for std::array checking.
    //  */
    // template <typename T>                  struct is_std_array : std::false_type { };
    // template <typename T, std::size_t DIM> struct is_std_array<std::array<T, DIM>> : std::true_type { };
    // template <typename T> concept std_array = is_std_array<T>::value;

    namespace enumerable  {
      namespace action {
      
	template<typename FUNCTION>
	concept linear_qfunction = requires() {
	  typename FUNCTION::state_type;
	  typename FUNCTION::params_type;
	  typename FUNCTION::action_type;
	  typename FUNCTION::state_feature_type;
	}	&&
	requires(FUNCTION f) {
	  {f.params}    -> std::same_as<std::shared_ptr<typename FUNCTION::params_type>&>;
	  {f.s_feature} -> std::same_as<std::shared_ptr<typename FUNCTION::state_feature_type>&>;
	} &&      
	nuplet<typename FUNCTION::params_type> &&
	finite<typename FUNCTION::action_type> &&
	feature<typename FUNCTION::state_feature_type, typename FUNCTION::state_type>;
      
      }
    }

    
  }
}
