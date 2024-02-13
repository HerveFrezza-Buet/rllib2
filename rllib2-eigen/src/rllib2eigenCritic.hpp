#pragma once

#include <rllib2.hpp>
#include <Eigen/Dense>

#include <rllib2eigenDiscreteA.hpp>

namespace rl2 {
  namespace eigen {
    namespace critic {
      namespace discrete_a {
	template<typename S_FEATURE, rl2::concepts::enumerable A, typename ITERATOR, typename SENTINEL>
	inline void lstd(S_FEATURE&& s_phi, ITERATOR begin, SENTINEL end) {
	}
      }
    }
  }
}
