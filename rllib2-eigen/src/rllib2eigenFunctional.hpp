#pragma once

#include <cmath>
#include <complex>
#include <array>
#include <initializer_list>
#include <Eigen/Dense>
#include <rllib2eigenConcepts.hpp>


namespace rl2 {
  namespace eigen {

    namespace function {

      template<typename DOMAIN>
      struct Gaussian {
      private:
	DOMAIN mu;
	double sigma;
	double gamma;

      public:
	
	Gaussian(const DOMAIN& mu, double sigma) : mu(mu), sigma(sigma), gamma(.5/(sigma*sigma)) {}
	Gaussian() : Gaussian({}, 1.) {}
	Gaussian(const Gaussian&) = default;
	Gaussian& operator=(const Gaussian&) = default;

	double operator()(const DOMAIN& x) const {
	  auto delta = x - mu;
	  return std::exp(-delta.dot(delta) * gamma);
	}
      };
      
      template<>
      struct Gaussian<double> {
      private:
	double mu;
	double sigma;
	double gamma;

      public:
	
	Gaussian(const double& mu, double sigma) : mu(mu), sigma(sigma), gamma(.5/(sigma*sigma)) {}
	Gaussian() : Gaussian(0, 1.) {}
	Gaussian(const Gaussian&) = default;
	Gaussian& operator=(const Gaussian&) = default;

	

	double operator()(const double& x) const {
	  auto delta = x - mu;
	  return std::exp(- delta * delta * gamma);
	}
      };

      template<typename DOMAIN>
      auto gaussian(const DOMAIN& mu, double sigma) {return Gaussian(mu, sigma);}
    }

    namespace feature {
      
      template<unsigned int DEGREE>
      struct polynomial {
      public:
	using return_type = Eigen::Vector<double, DEGREE+1>;
	
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
      
	return_type operator()(double x) const {
	  return_type res;
	  auto it = res.begin();
	  *it = 1;
	  fill<DEGREE>(it, x);
	  return res;
	}
      };

      template<typename DOMAIN, unsigned int NB_RBF>
      struct gaussian_rbf {
	using rbf_type = function::Gaussian<DOMAIN>;
	using return_type = Eigen::Vector<double, NB_RBF + 1>;
	
	std::array<function::Gaussian<DOMAIN>, NB_RBF> rbfs;
	
	gaussian_rbf() = default;
	gaussian_rbf(const gaussian_rbf&) = default;
	gaussian_rbf& operator=(const gaussian_rbf&) = default;

	// I would have preferred something like this.. but I do not know how to force size checking at compiling time.
	// gaussian_rbf(const std::initializer_list<rbf_type>& args) : rbfs(args) {}
	
	gaussian_rbf(const std::initializer_list<rbf_type>& args) : rbfs() {
	  auto it = args.begin();
	  for(auto& rbf : rbfs) rbf = *(it++);
	}
	
	return_type operator()(const DOMAIN& x) const {
	  return_type res;
	  auto res_it = res.begin();
	  *(res_it++) = 1; // This is the offset.
	  for(auto& rbf : rbfs) *(res_it++) = rbf(x);
	  return res;
	}
      };
    }

    
    namespace linear {

      // Consider providing a std::cref(....) expression as param argument.
      template<typename AMBIENT, unsigned int DIM, concepts::feature<AMBIENT, DIM> FEATURE, typename PARAM>
      auto make(const FEATURE& phi, PARAM param) {
	return [phi, param](const AMBIENT& arg) {
	  const Eigen::Vector<double, DIM>& p = param; // Required if param is a std::cref
	  return phi(arg).transpose() * p;
	};
      }
    }
  }
}
