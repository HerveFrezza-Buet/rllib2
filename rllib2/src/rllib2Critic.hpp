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
      }
      
      namespace enumerable {
	namespace action {
	  namespace bellman {
	    template<typename S, typename A, concepts::q_function<S, A> Q>
	    requires concepts::enumerable::finite<A>
	    double optimality(const Q& q, double gamma, const sarsa<S, A>& transition) {
	      auto [s, a, r, ss, aa] = transition;
	      return r + gamma * rl2::enumerable::algo::max<A>(q(ss));
	    }
	  }
	}
      }
	
      template<typename S, typename A, concepts::q_function<S, A> Q, concepts::bellman_operator<Q, S, A, sarsa<S, A>> BELLMAN_OP>
      double error(const Q& q, double gamma, const sarsa<S, A>& transition, const BELLMAN_OP& bellman_op) {
	return bellman_op(q, gamma, transition) - q(transition.s, transition.a);
      }
      
      
      template<typename S, typename A, concepts::q_function<S, A> Q>
      requires concepts::enumerable::two_args_function<Q>
      double update(Q& q, const S& s, const A& a, double alpha, double td_error) {
	auto delta = alpha * td_error;
	auto it = q.params_it + static_cast<std::size_t>(typename Q::arg_type(typename Q::first_entry_type(s), typename Q::second_entry_type(a)));
	*it += delta;
	return delta;
      }
      
    }
  }		  
}
