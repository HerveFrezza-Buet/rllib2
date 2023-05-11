#pragma once

#include <concepts>
#include <iterator>
#include <random>
#include <functional>
#include <array>
#include <memory>
#include <type_traits>

namespace rl2 {

  namespace discrete {
    /**
     * This function returns a function that provides a random value of
     * type TYPE at each call.
     */
    template<specs::enumerable TYPE, typename RANDOM_GENERATOR>
    auto uniform(RANDOM_GENERATOR& gen) {
      return [&gen]() -> TYPE {return std::uniform_int_distribution<std::size_t>(0, TYPE::size-1)(gen);};
    }


    /**
     * This makes an epsilon version of a function. It calls f with a
     * probability 1-epsilon, and return a random result otherwise.
     */
    template<typename F,
	     std::convertible_to<double> PARAM, // PARAM can be eps, std::ref(eps), std::cref(eps).
	     typename RANDOM_GENERATOR> 
    auto epsilon(const F& f, const PARAM& p, RANDOM_GENERATOR& gen) {
      return [&f, p, &gen](auto arg) -> auto {
	typename std::remove_const<typename std::remove_reference<decltype(f(arg))>::type>::type res;
	if(std::bernoulli_distribution(p)(gen))
	  res = std::uniform_int_distribution<std::size_t>(0, decltype(res)::size)(gen);
	else
	  res = f(arg);
	return res;
      };
    }

    /**
     * @return a function g such as g() = argmax_{x in S} f(x).
     */
    template<specs::enumerable S,
	     std::invocable<S> F,
	     typename COMP = std::less<std::invoke_result_t<F, S>>>
    auto greedy(const F& f) {
      return [&f, comp=COMP()]() -> S {
	auto it = S::begin;
	auto max_value = f(it); // it is implicitly casted into a S.
	auto argmax    = it++;
	for(; it != S::end; ++it) 
	  if(auto value = f(it); comp(max_value, value)) {
	    max_value = value;
	    argmax = it;
	  }
	return argmax;
      };
    }
	

  }

  namespace tabular {
    template<specs::enumerable X, std::random_access_iterator IT>
    struct function {
      using arg_type    = X;
      using return_type = std::iter_value_t<IT>;
      
      IT params_it; 

      function()                           = delete;
      function(const function&)            = default;
      function(function&&)                 = default;
      function& operator=(const function&) = default;
      function& operator=(function&&)      = default;
      function(IT params_it) : params_it(params_it) {}
      
      return_type operator()(std::size_t idx)                const {return *(params_it + idx                         );}
      return_type operator()(const X& x)                     const {return *(params_it + static_cast<std::size_t>(x ));}
      return_type operator()(const typename X::iterator& it) const {return *(params_it + static_cast<std::size_t>(it));}
    };

    template<specs::enumerable X, std::random_access_iterator IT>
    auto make_function(IT params_it) {return function<X, IT>(params_it);}
    

  }
  
}
