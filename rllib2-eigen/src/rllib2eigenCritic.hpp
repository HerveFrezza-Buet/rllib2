#pragma once

#include <rllib2.hpp>
#include <Eigen/Dense>
#include <utility>


namespace rl2 {
  namespace eigen {
    namespace critic {
      namespace discrete_a {

	template<rl2::concepts::discrete_a::linear_qfunction Q,
		 rl2::concepts::policy<typename Q::state_type, typename Q::action_type> POLICY,
		 std::input_iterator Iter, std::sentinel_for<Iter> Sentinel>
	void lstd(Q q, const POLICY& pi, Iter begin, Sentinel end) {
	  Eigen::Vector<double, Q::params_type::dim>& theta = *(q.params);
	  
	}
      }
    }
  }
}
