#pragma once

#include <Eigen/Dense>

namespace rl2 {
  namespace eigen {
    namespace concepts {

      template<typename AMBIENT, int DIM>
      concept feature =
	requires(const feature cf, const AMBIANT ca) {
	{cf(ca)} -> std::same_as<Eigen::Vector<double, DIM>>;
      };
    }
  }
}

