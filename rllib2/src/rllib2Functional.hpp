/*

Copyright 2024 Herve FREZZA-BUET, Alain DUTECH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#pragma once

#include <concepts>
#include <iterator>
#include <random>
#include <functional>
#include <array>
#include <memory>
#include <type_traits>
#include <utility>
#include <type_traits>

#include <rllib2Concepts.hpp>
#include <rllib2Enumerable.hpp>
#include <rllib2Nuplet.hpp>

namespace rl2 {

  namespace enumerable {
    /**
     * This function returns a function that provides a random value of
     * type TYPE at each call.
     */
    template<concepts::enumerable::finite TYPE, typename RANDOM_GENERATOR>
    auto uniform_sampler(RANDOM_GENERATOR& gen) {
      return [&gen]() -> TYPE {return std::uniform_int_distribution<std::size_t>(0, TYPE::size()-1)(gen);};
    }



    /**
     * This makes an epsilon version of a function. It calls f with a
     * probability 1-epsilon, and return a random result
     * otherwise. The return type of the function passed as argument
     * has to be enumerable.
     */
    template<typename F,
	     std::convertible_to<double> PARAM, // PARAM can be eps, std::ref(eps), std::cref(eps).
	     typename RANDOM_GENERATOR> 
    auto epsilon_ify(const F& f, const PARAM& p, RANDOM_GENERATOR& gen) {
      return [f, p, &gen](auto arg) -> concepts::enumerable::finite auto {
	typename std::remove_const<typename std::remove_reference<decltype(f(arg))>::type>::type res;
	if(std::bernoulli_distribution(p)(gen))
	  res = std::uniform_int_distribution<std::size_t>(0, decltype(res)::size())(gen);
	else
	  res = f(arg);
	return res;
      };
    }

    namespace action {
      template<typename S, concepts::enumerable::finite A, typename RANDOM_GENERATOR>
      auto random_policy(RANDOM_GENERATOR& gen) {
	return [sampler = enumerable::uniform_sampler<A>(gen)](const S&) {return sampler();};
      }
      
    }
  }

  namespace enumerable {

    namespace algo {
      /**
       * @return max_{x in S} f(x).
       */
      template<concepts::enumerable::finite S,
	       std::invocable<S> F,
	       typename COMP = std::less<std::invoke_result_t<F, S>>>
      auto max(const F& f) {
	COMP comp;
	auto it = S::begin();
	auto max_value = f(it); // it is implicitly casted into a S.
	for(; it != S::end(); ++it) 
	  if(auto value = f(it); comp(max_value, value))
	    max_value = value;
	return max_value;
      }
    
    
      /**
       * @return argmax_{x in S} f(x).
       */
      template<concepts::enumerable::finite S,
	       std::invocable<S> F,
	       typename COMP = std::less<std::invoke_result_t<F, S>>>
      S argmax(const F& f) {
	COMP comp;
	auto it = S::begin();
	auto max_value = f(it); // it is implicitly casted into a S.
	auto res        = it++;
	for(; it != S::end(); ++it) 
	  if(auto value = f(it); comp(max_value, value)) {
	    max_value = value;
	    res       = it;
	  }
	return res;
      }
    }
    
    /**
     * @short Makes a tabular function from an array of values
     */
    template<concepts::enumerable::finite X, std::random_access_iterator IT>
    struct tabular {
      using arg_type    = X;
      using result_type = std::iter_value_t<IT>;
      using params_iterator_type = IT;
      
      IT params_it; 

      tabular()                           = delete;
      tabular(const tabular&)            = default;
      tabular(tabular&&)                 = default;
      tabular& operator=(const tabular&) = default;
      tabular& operator=(tabular&&)      = default;
      tabular(IT params_it) : params_it(params_it) {}
      
      result_type operator()(std::size_t idx)                const {return *(params_it + idx                         );}
      result_type operator()(const X& x)                     const {return *(params_it + static_cast<std::size_t>(x ));}
      result_type operator()(const typename X::iterator& it) const {return *(params_it + static_cast<std::size_t>(it));}
    };

    template<concepts::enumerable::finite X, std::random_access_iterator IT>
    auto make_tabular(IT params_it) {return tabular<X, IT>(params_it);}

    /**
     * @short Makes a tabular taking 2 arguments.
     */
    template<concepts::enumerable::finite X, concepts::enumerable::finite Y, std::random_access_iterator IT>
    struct two_args_tabular : public tabular<enumerable::pair<X, Y>, IT> {
      using params_iterator_type = typename tabular<enumerable::pair<X, Y>, IT>::params_iterator_type;
      using arg_type             = typename tabular<enumerable::pair<X, Y>, IT>::arg_type;
      using result_type          = typename tabular<enumerable::pair<X, Y>, IT>::result_type;
      using first_entry_type  = X;
      using second_entry_type = Y;
      using tabular<enumerable::pair<X, Y>, IT>::tabular;
      using tabular<enumerable::pair<X, Y>, IT>::operator=;
      using tabular<enumerable::pair<X, Y>, IT>::operator();

      auto operator()(const X& x, const Y& y) const {return (*this)(arg_type(x, y));}
      auto operator()(const X& x)             const {return make_tabular<Y, IT>(this->params_it + Y::size() * static_cast<std::size_t>(x));}
    };

    template<concepts::enumerable::finite X, concepts::enumerable::finite Y, std::random_access_iterator IT>
    auto make_two_args_tabular(IT params_it) {return two_args_tabular<X, Y, IT>(params_it);}

    
    template<concepts::two_args_function TWO_ARGS_FUNCTION, typename COMP=std::less<typename TWO_ARGS_FUNCTION::result_type>>
    requires concepts::enumerable::finite<typename TWO_ARGS_FUNCTION::second_entry_type>
    auto argmax_ify(TWO_ARGS_FUNCTION taf) {
      return [taf](const typename TWO_ARGS_FUNCTION::first_entry_type& arg) {return algo::argmax<typename TWO_ARGS_FUNCTION::second_entry_type, decltype(taf(std::declval<typename TWO_ARGS_FUNCTION::first_entry_type>())), COMP>(taf(arg));};
    }
	
    template<concepts::two_args_function TWO_ARGS_FUNCTION, typename COMP=std::less<typename TWO_ARGS_FUNCTION::result_type>>
    auto greedy_ify(TWO_ARGS_FUNCTION taf) {return argmax_ify<TWO_ARGS_FUNCTION, COMP>(taf);}
  }

  namespace linear {
    
    namespace enumerable {
      namespace action {

      
	template<typename S, concepts::enumerable::finite A, concepts::feature<S> S_FEATURE>
	struct q_dim : std::integral_constant<std::size_t, S_FEATURE::dim * A::size()> {};
      
	template<typename S, concepts::enumerable::finite A, concepts::feature<S> S_FEATURE>
	inline constexpr std::size_t q_dim_v = q_dim<S, A, S_FEATURE>::value;
      
      
	template<concepts::nuplet PARAMS, typename S, concepts::enumerable::finite A, concepts::feature<S> S_FEATURE>
	requires (PARAMS::dim == q_dim_v<S, A, S_FEATURE>)
	struct q {
	  using params_type = PARAMS;
	  using state_type = S;
	  using action_type = A;
	  using state_feature_type = S_FEATURE;
	  constexpr static std::size_t dim = S_FEATURE::dim * A::size();
	
	  using result_type = double;
	  using first_entry_type = state_type;
	  using second_entry_type = action_type;

	  std::shared_ptr<state_feature_type> s_feature;
	  std::shared_ptr<params_type> params;

	  double operator()(const S& s, const A& a) const { 
	    auto it = params->begin();
	    std::advance(it, static_cast<std::size_t>(a) * S_FEATURE::dim);
	    double res = 0;
	    for(auto phi_coef : (*s_feature)(s)) res += phi_coef * *(it++);
	    return res;
	  }

	  auto operator()(const S& s) const {
	    return [*this, s](const A& a) { // We capture this by value, since it is only a pair of shared pointers.
	      return (*this)(s, a);
	    };
	  }
	
	      
	};
      }
    }
  }
  
}
