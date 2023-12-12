#pragma once

#include <Eigen/Dense>
#include <rllib2eigenConcepts.hpp>

namespace rl2 {
  namespace eigen {

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
    }

    
    namespace linear {

      // Consider providing a std::cref(....) expression as param argument.
      template<typename AMBIENT, int DIM, concepts::feature<AMBIENT, DIM> FEATURE, typename PARAM>
      auto make(const FEATURE& phi, PARAM param) {
	return [phi, param](const AMBIENT& arg) {
	  const Eigen::Vector<double, DIM>& p = param; // Required if param is a std::cref
	  return phi(arg).transpose() * p;
	};
      }
    }
  }
}
