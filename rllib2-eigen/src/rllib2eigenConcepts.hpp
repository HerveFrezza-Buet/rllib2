#pragma once

#include <concepts>
#include <Eigen/Dense>

namespace rl2 {
  namespace eigen {
    namespace concepts {

      template<typename FEATURE, typename AMBIENT, unsigned int DIM>
      concept feature =
	std::copy_constructible<FEATURE>
	&& FEATURE::dim == DIM
	&& requires(const FEATURE cf, const AMBIENT ca) {
	typename FEATURE::ambient_type;
	{cf(ca)} -> std::same_as<Eigen::Vector<double, DIM>>;
      };


      
    }
  }
}

