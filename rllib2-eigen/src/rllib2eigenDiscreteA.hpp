#pragma once

#include <tuple>
#include <rllib2.hpp>
#include <rllib2egenConcepts.hpp>

namespace rl2 {
  namespace eigen {
    namespace feature {
      namespace descrete_a {


	// c'est complètement con, il faut un seul phi pour les actions... c'est un linear qu'il faut fairen à partir d'un seul phi sur les états.
	template<typename S, rllib2::concepts::enumerable A, unsigned int S_DIM, rllib2::eigen::concepts::feature<S, S_DIM> S_FEATURE>
	struct phi {
	  constexpr static unsigned int dim = S_DIM * A::size();
	  using ambient_type = std::pair<S, A>;
	  using return_type = Eigen::Vector<double, dim>;
	  
	  phi() = default;
	  phi(const phi&) = default;
	  phi& operator=(const phi&) = default;
	  phi(phi&&) = default;
	  phi& operator=(phi&&) = default;
	  
	  return_type operator()(const ambient_type& sa) const {
	    return_type res;
	    auto it = res.begin();
	    
	    *it = 1;
	    fill<DEGREE>(it, x);
	    return res;
	  }
	};
      }
    }
  }
}
