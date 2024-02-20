#pragma once

#include <cstddef>
#include <ranges>
#include <cmath>

#include <gdyn.hpp>
#include <rllib2Functional.hpp>

namespace rl2 {

  namespace functional {
    template<typename INPUT>
    struct gaussian {};

    template<>
    struct gaussian<double> {
      double mu;
      double gamma;
      
      gaussian(double mu, double sigma) : mu(mu), gamma(.5/(sigma*sigma)) {}
      gaussian() : gaussian(0, 1.) {}
      gaussian(const gaussian&) = default;
      gaussian& operator=(const gaussian&) = default;
   
      double operator()(const double& x) const {
	auto delta = x - mu;
	return std::exp(- delta * delta * gamma);
      }
    };
  }
    
  namespace features {
    
    template<unsigned int DEGREE>
    struct polynomial {
      constexpr static std::size_t dim = DEGREE + 1;
      auto operator()(double x) const {
	return linear::make_params_from_range<dim>(gdyn::views::pulse([x, next = 1]() mutable {auto res = next; next *= x; return res;})
						   | std::views::take(dim));
      }
    };

    template<unsigned int NB_RBF>
    struct gaussian_rbf {
      constexpr static std::size_t dim = DEGREE + 1;
    }

  }
}
