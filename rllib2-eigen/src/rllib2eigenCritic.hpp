#pragma once

#include <rllib2.hpp>
#include <Eigen/Dense>
#include <utility>


namespace rl2 {
  namespace eigen {
    namespace critic {
      namespace discrete_a {

	template<typename Q, typename POLICY, typename Iter, typename Sentinel>
	void lstd(Q q, const POLICY& pi, Iter begin, Sentinel end) {
	  
	}
      }
    }
  }
}
