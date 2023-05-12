#pragma once

#include <rllib2Transition.hpp>
#include <rllib2Functional.hpp>

namespace rl2 {
  namespace critic {
    namespace td {
      
      template<typename S, typename A, specs::q_function<S, A> Q>
      double evaluation_error(const Q& q, double gamma, const sarsa<S, A>& transition) {
	auto [s, a, r, ss, aa] = transition;
	if(aa)
	  return r + gamma * q(ss, *aa) - q(s, a);
	else
	  return r - q(s, a);
      }
      
      namespace discrete {
	template<typename S, typename A, specs::q_function<S, A> Q>
	requires specs::enumerable<A>
	double optimal_error(const Q& q, double gamma, const sarsa<S, A>& transition) {
	  auto [s, a, r, ss, aa] = transition;
	  return r + gamma * algo::argmax<A>(q(ss)) - q(s, a);
	}
      }
      
      template<typename S, typename A, specs::q_function<S, A> Q>
      requires specs::tabular_table<Q>
      void update(Q& q, const S& s, const A& a, double td_error, double alpha) {
	auto it = q.params_it + static_cast<std::size_t>(typename Q::arg_type(typename Q::first_type(s), typename Q::first_type(a)));
	*it += alpha * (error - *it);
      }
      
    }
  }		  
}
