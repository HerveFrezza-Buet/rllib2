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
	
      template<typename S, typename A, concepts::q_function<S, A> Q, concepts::bellman::q_update<Q, S, A, sarsa<S, A>> BELLMAN_OP>
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

    namespace dp {
      namespace enumerable {

	
	template<concepts::enumerable::finite S,
		 concepts::enumerable::finite A,
		 concepts::transition_distribution<S, A> T,
		 concepts::reward_function<S, A> R,
		 concepts::value_function<S> V,
		 concepts::assignable_function<S, A> POLICY>
	void greedy_policy_from_values(POLICY& pi, const T& t, const R& r, const V& v, double gamma) {
	  for(auto s_it = S::begin(); s_it != S::end(); ++s_it) {
	    auto s = *s_it;
	    A argmax_a;
	    double max_sum = std::numeric_limits<double>::lowest();
	    for(auto a_it = A::begin(); a_it !=  A::end(); ++a_it) {
	      double sub_sum = 0;
	      auto a = *a_it;
	      for(auto ss_it = S::begin(); ss_it != S::end(); ++ss_it) {
		auto ss = *ss_it;
		sub_sum += t(s, a, ss) * (r(s, a, ss) + gamma * v(ss));
	      }
	      if(sub_sum > max_sum) {
		max_sum = sub_sum;
		argmax_a = a;
	      }
	    }
	    pi(s) = argmax_a;
	  }
	}
	

	template<concepts::enumerable::finite S,
		 concepts::enumerable::finite A,
		 concepts::transition_distribution<S, A> T,
		 concepts::reward_function<S, A> R,
		 concepts::assignable_function<S, double> V>
	void value_iteration(V& v, const T& t, const R& r, double gamma, double epsilon) {
	  double delta;
	  do {
	    delta = 0;
	    for(auto s_it = S::begin(); s_it != S::end(); ++s_it) {
	      auto s = *s_it;
	      double value = v(s);
	      double new_value = std::numeric_limits<double>::lowest();
	      for(auto a_it = A::begin(); a_it != A::end(); ++a_it) {
		double sub_sum = 0;
		auto a = *a_it;
		for(auto ss_it = S::begin(); ss_it != S::end(); ++ss_it) {
		  auto ss = *ss_it;
		  sub_sum += t(s, a, ss) * (r(s, a, ss) + gamma * v(ss));
		}
		if(sub_sum > new_value) new_value = sub_sum;
	      }
	      v(s) = new_value;
	      delta = std::max(delta, std::fabs(value - new_value));
	    }
	  } while(delta > epsilon);
	}

	
      }
    }
    
    
  }		  
}
