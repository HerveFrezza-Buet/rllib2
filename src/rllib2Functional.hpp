#pragma once

#include <concepts>
#include <iterator>
#include <random>
#include <functional>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>

#include <rllib2Concepts.hpp>
#include <rllib2Enumerable.hpp>

namespace rl2 {

  namespace discrete {
    /**
     * This function returns a function that provides a random value of
     * type TYPE at each call.
     */
    template<concepts::enumerable TYPE, typename RANDOM_GENERATOR>
    auto uniform_sampler(RANDOM_GENERATOR& gen) {
      return [&gen]() -> TYPE {return std::uniform_int_distribution<std::size_t>(0, TYPE::size-1)(gen);};
    }


    /**
     * This makes an epsilon version of a function. It calls f with a
     * probability 1-epsilon, and return a random result
     * otherwise. The return type of the function passed as argument
     * has to beenumerable.
     */
    template<typename F,
	     std::convertible_to<double> PARAM, // PARAM can be eps, std::ref(eps), std::cref(eps).
	     typename RANDOM_GENERATOR> 
    auto epsilon_ify(const F& f, const PARAM& p, RANDOM_GENERATOR& gen) {
      return [f, p, &gen](auto arg) -> concepts::enumerable auto {
	typename std::remove_const<typename std::remove_reference<decltype(f(arg))>::type>::type res;
	if(std::bernoulli_distribution(p)(gen))
	  res = std::uniform_int_distribution<std::size_t>(0, decltype(res)::size)(gen);
	else
	  res = f(arg);
	return res;
      };
    }

    namespace algo {
      /**
       * @return max_{x in S} f(x).
       */
      template<concepts::enumerable S,
	       std::invocable<S> F,
	       typename COMP = std::less<std::invoke_result_t<F, S>>>
      auto max(const F& f) {
	COMP comp;
	auto it = S::begin;
	auto max_value = f(it); // it is implicitly casted into a S.
	for(; it != S::end; ++it) 
	  if(auto value = f(it); comp(max_value, value))
	    max_value = value;
	return max_value;
      }
    
    
      /**
       * @return argmax_{x in S} f(x).
       */
      template<concepts::enumerable S,
	       std::invocable<S> F,
	       typename COMP = std::less<std::invoke_result_t<F, S>>>
      S argmax(const F& f) {
	COMP comp;
	auto it = S::begin;
	auto max_value = f(it); // it is implicitly casted into a S.
	auto res        = it++;
	for(; it != S::end; ++it) 
	  if(auto value = f(it); comp(max_value, value)) {
	    max_value = value;
	    res       = it;
	  }
	return res;
      }
    }
	
  }

  namespace tabular {
    /**
     * @short Makes a function from an array of values
     */
    template<concepts::enumerable X, std::random_access_iterator IT>
    struct function {
      using arg_type    = X;
      using result_type = std::iter_value_t<IT>;
      using params_iterator_type = IT;
      
      IT params_it; 

      function()                           = delete;
      function(const function&)            = default;
      function(function&&)                 = default;
      function& operator=(const function&) = default;
      function& operator=(function&&)      = default;
      function(IT params_it) : params_it(params_it) {}
      
      result_type operator()(std::size_t idx)                const {return *(params_it + idx                         );}
      result_type operator()(const X& x)                     const {return *(params_it + static_cast<std::size_t>(x ));}
      result_type operator()(const typename X::iterator& it) const {return *(params_it + static_cast<std::size_t>(it));}
    };

    template<concepts::enumerable X, std::random_access_iterator IT>
    auto make_function(IT params_it) {return function<X, IT>(params_it);}

    /**
     * @short Makes a function taking 2 arguments.
     */
    template<concepts::enumerable X, concepts::enumerable Y, std::random_access_iterator IT>
    struct two_args_function : public function<enumerable::pair<X, Y>, IT> {
      using params_iterator_type = typename function<enumerable::pair<X, Y>, IT>::params_iterator_type;
      using arg_type             = typename function<enumerable::pair<X, Y>, IT>::arg_type;
      using result_type          = typename function<enumerable::pair<X, Y>, IT>::result_type;
      using first_entry_type  = X;
      using second_entry_type = Y;
      using function<enumerable::pair<X, Y>, IT>::function;
      using function<enumerable::pair<X, Y>, IT>::operator=;
      using function<enumerable::pair<X, Y>, IT>::operator();

      auto operator()(const X& x, const Y& y) const {return (*this)(arg_type(x, y));}
      auto operator()(const X& x)             const {return make_function<Y, IT>(this->params_it + Y::size * static_cast<std::size_t>(x));}
    };

    template<concepts::enumerable X, concepts::enumerable Y, std::random_access_iterator IT>
    auto make_two_args_function(IT params_it) {return two_args_function<X, Y, IT>(params_it);}
  }

  namespace discrete {
    template<concepts::two_args_function TWO_ARGS_FUNCTION, typename COMP=std::less<typename TWO_ARGS_FUNCTION::result_type>>
    requires concepts::enumerable<typename TWO_ARGS_FUNCTION::second_entry_type>
    auto argmax_ify(TWO_ARGS_FUNCTION taf) {
      return [taf](const typename TWO_ARGS_FUNCTION::first_entry_type& arg) {return algo::argmax<typename TWO_ARGS_FUNCTION::second_entry_type, decltype(taf(std::declval<typename TWO_ARGS_FUNCTION::first_entry_type>())), COMP>(taf(arg));};
    }
	
    template<concepts::two_args_function TWO_ARGS_FUNCTION, typename COMP=std::less<typename TWO_ARGS_FUNCTION::result_type>>
    auto greedy_ify(TWO_ARGS_FUNCTION taf) {return argmax_ify<TWO_ARGS_FUNCTION, COMP>(taf);}
  }
  
}
