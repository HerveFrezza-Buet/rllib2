#pragma once

#include <tuple>
#include <rllib2.hpp>
#include <rllib2eigenConcepts.hpp>

namespace rl2 {
  namespace eigen {
    namespace feature {
      namespace discrete_a {
	
	template<typename S, rl2::concepts::enumerable A, unsigned int S_DIM, rl2::eigen::concepts::feature<S, S_DIM> S_FEATURE>
	struct phi {
	  constexpr static unsigned int dim = S_DIM * A::size();
	  using ambient_type = std::pair<S, A>;
	  using return_type = Eigen::Vector<double, dim>;
	  S_FEATURE s_phi;
	  
	  phi() = default;
	  phi(const phi&) = default;
	  phi& operator=(const phi&) = default;
	  phi(phi&&) = default;
	  phi& operator=(phi&&) = default;
	  phi(const S_FEATURE& s_phi) : s_phi(s_phi) {}
	  phi(S_FEATURE&& s_phi) : s_phi(std::move(s_phi)) {}
	  
	  return_type operator()(const ambient_type& sa) const {
	    return_type res;
	    res.segment(static_cast<std::size_t>(std::get<1>(sa)) * S_DIM, S_DIM) = s_phi(std::get<0>(sa));
	    return res;
	  }
	};

	template<typename S, rl2::concepts::enumerable A, unsigned int S_DIM, rl2::eigen::concepts::feature<S, S_DIM> S_FEATURE>
	struct linear {
	  using param_type = Eigen::Vector<double, S_DIM * A::size()>;
	  S_FEATURE s_phi;
	  param_type theta;
	  
	  linear() = delete;
	  linear(const linear&) = default;
	  linear& operator=(const linear&) = default;
	  linear(const S_FEATURE& s_phi) : s_phi(s_phi), theta() {}
	  linear(S_FEATURE&& s_phi) : s_phi(std::move(s_phi)), theta() {}
	  
	  double operator()(const S& s, const A& a) const {
	    return s_phi(s).transpose() * theta.segment(static_cast<std::size_t>(a) * S_DIM, S_DIM);
	  }

	  double operator()(const std::pair<S, A>& sa) const {
	    return (*this)(std::get<0>(sa), std::get<1>(sa));
	  }
	  
	};
		      


      }
    }
  }
}
