#pragma once

#include <cstddef>
#include <ranges>
#include <cmath>
#include <array>
#include <optional>
#include <memory>
#include <iostream>
#include <tuple>

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
      auto it = sigmas.begin();
      for(auto& g : gammas) g = gaussian_gamma_of_sigma(*it++);
      return gammas;
    }
    
    template<typename MU>
    requires (concepts::nuplet<MU> || std::same_as<MU, double>)
    struct gaussian;
    
    template<concepts::nuplet MU>
    struct gaussian<MU> {
      MU mu;
      std::shared_ptr<MU> gammas_ptr;
      
      template<typename X, concepts::nuplet_wrapper<X> WRAPPER = nuplet::by_default::wrapper<X>>
      double operator()(const X& x) const {
	double sum = 0;
	auto mu_it = mu.begin();
	auto gamma_it = gammas_ptr->begin();
	WRAPPER wrapper {x};
	for(auto x_it = wrapper.begin(); x_it != wrapper.end(); ) {
	  auto delta = *(mu_it++) - *(x_it++);
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
   
      template<typename X, typename WRAPPER>
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
    struct rbfs {
      constexpr static std::size_t nb_rbfs = NB_RBFS;
      constexpr static std::size_t dim = NB_RBFS + 1;
      using rbfs_type = std::array<RBF, NB_RBFS>;
      std::shared_ptr<rbfs_type> rbfs;
      
      template<typename X, typename WRAPPER = nuplet::by_default::wrapper<X>>
      auto operator()(X&& x) const {
	return nuplet::make_from_range<dim>(gdyn::views::pulse([x = std::forward<X>(x), it = rbfs->end(), this]() mutable -> double {
	  if(it == rbfs->end()) {
	    it = rbfs->begin();
	    return 1;
	  }
	  else
	    return (*(it++)).template operator()<X, WRAPPER>(x);
	})
	  | std::views::take(dim));
      }

      
    };

  }
}
