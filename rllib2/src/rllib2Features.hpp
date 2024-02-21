#pragma once

#include <cstddef>
#include <ranges>
#include <cmath>
#include <array>
#include <optional>
#include <memory>
#include <iostream>

#include <gdyn.hpp>
#include <rllib2Functional.hpp>
#include <rllib2Nuplet.hpp>
#include <rllib2Concepts.hpp>

namespace rl2 {

  namespace functional {

    inline double gaussian_gamma_of_sigma(double sigma) {return .5/(sigma*sigma);}
    
    template<concepts::nuplet X>
    X gaussian_gammas_of_sigmas(const X& sigmas) {
      X gammas;
      auto it = gammas.begin();
      for(auto& g : gammas) g = gaussian_gamma_of_sigma(*it++);
      return gammas;
    }
    
    template<typename X>
    struct gaussian;
    
    template<concepts::nuplet X>
    struct gaussian<X> {
      X mu;
      std::shared_ptr<X> gammas_ptr;
      
      double operator()(const X& x) const {
	double sum = 0;
	auto mu_it = mu.begin();
	auto gamma_it = gammas_ptr->begin();
	for(auto x_comp : x) {
	  auto delta = *(mu_it++) - x;
	  sum += *(gamma_it++) * delta * delta;
	}
	return std::exp(-sum);
      }
    };

    template<>
    struct gaussian<double> {
      double mu;
      double gamma;
      
      gaussian(double mu, double sigma) : mu(mu), gamma(gaussian_gamma_of_sigma(sigma)) {}
      gaussian() : gaussian(0, 1.) {}
      gaussian(const gaussian&) = default;
      gaussian& operator=(const gaussian&) = default;
   
      double operator()(const double& x) const {
	auto delta = x - mu;
	return std::exp(- delta * delta * gamma);
      }
    };

    inline std::ostream& operator<<(std::ostream& os, const gaussian<double>& g) {
      return  os << "gauss(mu = " << g.mu << ", gamma = " << g.gamma << ')';
    }
  }
    
  namespace features {
    
    template<unsigned int DEGREE>
    struct polynomial {
      constexpr static std::size_t dim = DEGREE + 1;
      auto operator()(double x) const {
	return nuplet::make_from_range<dim>(gdyn::views::pulse([x, res = std::optional<double>()]() mutable {
	  if(res) res = *res * x;
	  else    res = 1;
	  return *res;})
	  | std::views::take(dim));
      }
    };

    template<unsigned int NB_RBFS, typename RBF>
    struct rbf {
      constexpr static std::size_t nb_rbfs = NB_RBFS;
      constexpr static std::size_t dim = NB_RBFS + 1;
      using rbfs_type = std::array<RBF, NB_RBFS>;
      std::shared_ptr<rbfs_type> rbfs;

      template<typename X>
      auto operator()(X&& x) const {
	return nuplet::make_from_range<dim>(gdyn::views::pulse([mu = std::forward<X>(x), it = rbfs->end(), this]() mutable -> double {
	  if(it == rbfs->end()) {
	    it = rbfs->begin();
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
