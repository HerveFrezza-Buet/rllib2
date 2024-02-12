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

      
      template<typename FUNCTION, typename FEATURE, typename AMBIENT, unsigned int DIM>
      concept linear_function =
	feature<FEATURE, AMBIENT, DIM>
	&& requires(const FUNCTION cf, FUNCTION f, const AMBIENT ca) {
	typename FUNCTION::param_type;
	{f.phi} -> std::same_as<FEATURE&>;
	{f.theta} -> std::same_as<Eigen::Vector<double, DIM>&>;
	{cf(ca)} -> std::same_as<double>;
      };
	
      


      
    }
  }
}

