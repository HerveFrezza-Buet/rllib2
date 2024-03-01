#pragma once

#include <Eigen/Dense>

namespace rl2 {
  namespace eigen {
    namespace nuplet {
    
      template<std::size_t DIM>
      struct from : public Eigen::Vector<double, DIM> {
	constexpr static std::size_t dim = DIM;
      };
    
      
    }
  }
}
