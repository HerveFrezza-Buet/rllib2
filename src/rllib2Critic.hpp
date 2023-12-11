#pragma once

#include <rllib2Transition.hpp>
#include <rllib2Functional.hpp>

namespace rl2 {
  namespace critic {
    namespace td {

      
      namespace bellman {
	template<typename S, typename A, concepts::q_function<S, A> Q>
	double evaluation(const Q& q, double gamma, const sarsa<S, A>& transition) {
	  auto [s, a, r, ss, aa] = transition;
	  if(aa)
	    return r + gamma * q(ss, *aa);
	  else
	    return r;
	}
      
      
      
      namespace discrete {
	namespace bellman {
	  template<typename S, typename A, concepts::q_function<S, A> Q>
	  requires concepts::enumerable<A>
	  double optimality(const Q& q, double gamma, const sarsa<S, A>& transition) {
	    auto [s, a, r, ss, aa] = transition;
	    return r + gamma * rl2::discrete::algo::max<A>(q(ss));
	  }
	}
      }
	
      template<typename S, typename A, concepts::q_function<S, A> Q, concepts::bellman_operator<Q, S, A> BELLMAN_OP>
      double error(const Q& q, double gamma, const sarsa<S, A>& transition, const BELLMAN_OP& bellman_op) {
	return bellman_op(q, gamma, transition) - q(s, a);
      }
      
      
      template<typename S, typename A, concepts::q_function<S, A> Q>
      requires concepts::tabular_two_args_function<Q>
      double update(Q& q, const S& s, const A& a, double alpha, double td_error) {
	delta = alpha * td_error
	auto it = q.params_it + static_cast<std::size_t>(typename Q::arg_type(typename Q::first_entry_type(s), typename Q::second_entry_type(a)));
	*it += delta;
	return delta;
      }
      
    }
  }		  
}
