#pragma once

#include <concepts>
#include <random>
#include <functional>

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
    template<specs::enumerable_producer F,
	     std::convertible_to<double> PARAM, // PARAM can be eps, std::ref(eps), std::cref(eps).
	     typename RANDOM_GENERATOR> 
    auto epsilon(const F& f, const PARAM& p, RANDOM_GENERATOR& gen) {
      return [&f, p, &gen]() -> decltype(f()) {
	if(std::bernoulli_distribution(p)(gen))
	  return std::uniform_int_distribution<std::size_t>(0, decltype(f())::size-1)(gen);
	else
	  return f();
      };
    }

    /**
     * The returns g such as g() = argmax_{x in S} f(x).
     */
    template<specs::enumerable S,
	     std::invocable<S> F,
	     typename COMP = std::less<std::invoke_result_t<F, S>>>
    auto greedy(const F& f) {
      return [&f]() -> S {
	COMP comp;
	auto it = S::begin;
	auto max_value = f(*it);
	auto argmax    = it++;
	for(; it != S::end; ++it)
	  if(auto value = f(*it); comp(max_value, value)) {
	    max_value = value;
	    argmax = it;
	  }
	return it;
      };
    }
    
  }
  
}
