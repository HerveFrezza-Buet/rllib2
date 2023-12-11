#pragma once

#include <rllib2eigenConcepts.hpp>

namespace rl2 {
  namespace eigen {
    namespace linear {
      
      template<typename AMBIENT, int DIM, typename PARAM, concepts::feature FEATURE>
      auto make(const FEATURE& feature, PARAM param) {
	return [&feature](const AMBIENT& arg) {
	  return feature(arg).transpose() * param;
	};
      }

      // TO DO : rbf, (0, x, x², ...)
    }
  }
}
