#pragma once

#include <cstddef>
#include <ranges>
#include <cmath>
#include <array>
#include <optional>

#include <gdyn.hpp>
#include <rllib2Functional.hpp>
#include <rllib2Nuplet.hpp>

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
	return nuplet::from_range<dim>(gdyn::views::pulse([x, res = std::optional<double>()]() mutable {
	  if(res) res = *res * x;
	  else    res = 1;
	  return *res;})
	  | std::views::take(dim));
      }
    };

    template<unsigned int NB_RBFS, typename RBF>
    struct rbf {
      constexpr static std::size_t dim = NB_RBFS + 1;
      std::shared_ptr<std::array<RBF, NB_RBFS>> rbfs;

      template<typename X>
      auto operator()(X&& x) const {
	return nuplet::from_range<dim>(gdyn::views::pulse([mu = std::forward<X>(x), it = rbfs.end(), this]() mutable {
	  if(it == rbfs.end()) {
	    it = rbfs.begin();
	    return 1;
	  }
	  else
	    return (*(it++))(mu);
	})
	  | std::views::take(dim));
      }
    };

  }
}
