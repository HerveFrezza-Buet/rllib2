#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <utility>
#include <type_traits>
#include <initializer_list>
#include <Eigen/Dense>
#include <rllib2eigenConcepts.hpp>
#include <memory>


namespace rl2 {
  namespace eigen {

    namespace function {

      template<typename AMBIENT>
      struct gaussian {
	using gammas_type     = std::remove_cvref_t<Eigen::ArrayWrapper<AMBIENT>>;
	using gammas_ptr_type = std::shared_ptr<gammas_type>;
	
	AMBIENT mu;
	gammas_ptr_type gammas_ptr;

      public:

	gaussian(const AMBIENT& mu, gammas_ptr_type gammas_ptr) : mu(mu), gammas_ptr(gammas_ptr) {}
	gaussian() = default;
	gaussian(const gaussian&) = default;
	gaussian& operator=(const gaussian&) = default;

	double operator()(const AMBIENT& x) const {
	  auto delta = (x - mu).array().square() * (*gammas_ptr);
	  
	  return std::exp(-delta.sum());
	}
      };

      template<typename AMBIENT>
      auto gammas(const AMBIENT& sigmas) {
	typename gaussian<AMBIENT>::gammas_type x {sigmas.array().inverse() * .5};
	return std::make_shared<typename gaussian<AMBIENT>::gammas_type>(sigmas.array().inverse() * .5);
      }
	
      
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

    namespace feature {
      
      template<unsigned int DEGREE>
      struct polynomial {
      public:
	constexpr static unsigned int dim = DEGREE + 1;
	using return_type = Eigen::Vector<double, dim>;
	using ambient_type = double;
	
      private:

	// Could be more efficient without iiterating on the powers.
	template<unsigned int NB, typename Iter>
	static void fill(Iter out, double x) {
	  if constexpr(NB == 0) return;
	  else {
	    double val = *(out++) * x;
	    *out = val;
	    fill<NB-1>(out, x);
	  }
	}
	
      public:
      
	polynomial() = default;
	polynomial(const polynomial&) = default;
	polynomial& operator=(const polynomial&) = default;
	polynomial(polynomial&&) = default;
	polynomial& operator=(polynomial&&) = default;
	
	return_type operator()(double x) const {
	  return_type res;
	  auto it = res.begin();
	  *it = 1;
	  fill<DEGREE>(it, x);
	  return res;
	}
      };

      template<typename AMBIENT, unsigned int NB_RBF>
      struct gaussian_rbf {
	constexpr static unsigned int dim = NB_RBF + 1;
	using ambient_type = AMBIENT;
	using rbf_type = function::gaussian<AMBIENT>;
	using return_type = Eigen::Vector<double, dim>;
	
	std::array<function::gaussian<AMBIENT>, NB_RBF> rbfs;
	
	gaussian_rbf() = default;
	gaussian_rbf(const gaussian_rbf&) = default;
	gaussian_rbf& operator=(const gaussian_rbf&) = default;
	gaussian_rbf(gaussian_rbf&&) = default;
	gaussian_rbf& operator=(gaussian_rbf&&) = default;
	
	gaussian_rbf(const std::initializer_list<rbf_type>& args) : rbfs(args) {}
	
	return_type operator()(const AMBIENT& x) const {
	  return_type res;
	  auto res_it = res.begin();
	  *(res_it++) = 1; // This is the offset.
	  for(auto& rbf : rbfs) *(res_it++) = rbf(x);
	  return res;
	}
      };
    }

    namespace function {

      template<typename AMBIENT, unsigned int DIM, concepts::feature<AMBIENT, DIM> FEATURE>
      struct linear {
	using param_type = Eigen::Vector<double, DIM>;
	FEATURE phi;
	param_type theta;
	
	linear() = delete;
	linear(const linear&) = default;
	linear& operator=(const linear&) = default;
	linear(const FEATURE& phi) : phi(phi), theta() {}
	linear(FEATURE&& phi) : phi(std::move(phi)), theta() {}

	double operator()(const AMBIENT& arg) const {
	  return phi(arg).transpose() * theta;
	}
      };

      
      template<typename FEATURE>
      auto make_linear(FEATURE&& phi) {return linear<typename std::remove_cvref_t<FEATURE>::ambient_type,
						     std::remove_cvref_t<FEATURE>::dim,
						     std::remove_cvref_t<FEATURE>>(std::forward<FEATURE>(phi));}
    }


    
  }
}
