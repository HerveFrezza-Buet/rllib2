#pragma once

#include <concepts>
#include <random>
#include <functional>
#include <array>
#include <memory>

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
      return [&f, p, &gen](auto arg) -> decltype(f(arg)) {
	if(std::bernoulli_distribution(p)(gen))
	  return std::uniform_int_distribution<std::size_t>(0, decltype(f(arg))::size-1)(gen);
	else
	  return f(arg);
      };
    }

    /**
     * @return argmax_{x in S} f(x).
     */
    template<specs::enumerable S,
	     std::invocable<S> F,
	     typename COMP = std::less<std::invoke_result_t<F, S>>>
    S argmax(const F& f) {
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
    }

  }

  namespace tabular {
    template<specs::enumerable X, typename Y>
    struct function {
      using arg_type = X;
      using return_type = Y;
      using data_type = std::array<Y, X::size>;
      
      std::unique_ptr<data_type> data; // We store the content in the heap.

      function() : data(std::make_unique<data_type>()) {}
      function(const function&)            = default;
      function(function&&)                 = default;
      function& operator=(const function&) = default;
      function& operator=(function&&)      = default;
      
      const Y& operator()(std::size_t idx) const {return (*data)[idx];}
      Y& operator()(std::size_t idx) {return (*data)[idx];}

      const Y& operator()(const X& x) const {return (*data)[static_cast<std::size_t>(x)];}
      Y& operator()(const X& x) {return (*data)[static_cast<std::size_t>(x)];}
      
      const Y& operator()(const typename X::iterator& it) const {return (*data)[static_cast<std::size_t>(it)];}
      Y& operator()(const typename X::iterator& it) {return (*data)[static_cast<std::size_t>(it)];}

    };

      
    // template <specs::enumerable S, specs::enumerable A>
    // struct Q : public function<S, function<A, double>> {
    //   using function<S, function<A, double>>::function;
    //   using function<S, function<A, double>>::operator=;
    //   double  operator()(const S& s, const A& a) const {return (*this)(s)(a);}
    //   double& operator()(const S& s, const A& a)       {return (*this)(s)(a);}
    // };
  }
  
}
