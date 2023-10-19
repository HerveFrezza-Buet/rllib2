#pragma once

#include <rllib2Transition.hpp>
#include <rllib2Functional.hpp>

namespace rl2 {
  namespace critic {
    namespace td {
      
      template<typename S, typename A, concepts::q_function<S, A> Q>
      double evaluation_error(const Q& q, double gamma, const sarsa<S, A>& transition) {
	auto [s, a, r, ss, aa] = transition;
	if(aa)
	  return r + gamma * q(ss, *aa) - q(s, a);
	else
	  return r - q(s, a);
      }
      
      namespace discrete {
	template<typename S, typename A, concepts::q_function<S, A> Q>
	requires concepts::enumerable<A>
	double optimal_error(const Q& q, double gamma, const sarsa<S, A>& transition) {
	  auto [s, a, r, ss, aa] = transition;
	  return r + gamma * rl2::discrete::algo::max<A>(q(ss)) - q(s, a);
	}
      }
      
      template<typename S, typename A, concepts::q_function<S, A> Q>
      requires concepts::tabular_two_args_function<Q>
      void update(Q& q, const S& s, const A& a, double alpha, double td_error) {
	auto it = q.params_it + static_cast<std::size_t>(typename Q::arg_type(typename Q::first_entry_type(s), typename Q::second_entry_type(a)));
	*it += alpha * td_error;
      }
      
    }
  }		  
}
