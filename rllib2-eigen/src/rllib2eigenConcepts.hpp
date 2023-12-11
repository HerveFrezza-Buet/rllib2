#pragma once

#include <Eigen/Dense>

namespace rl2 {
  namespace eigen {
    namespace concepts {

      template<typename FEATURE, typename AMBIENT, int DIM>
      concept feature =
	requires(const FEATURE cf, const AMBIENT ca) {
	{cf(ca)} -> std::same_as<Eigen::Vector<double, DIM>>;
      };
    }
  }
}

