#pragma once

#include <rllib2eigenFunctional.hpp>

namespace rl2 {
  namespace eigen {
    namespace checkings {

      using polynomial_feature = feature::polynomial<10>;
      static_assert(concepts::feature<polynomial_feature, double, 11>);

      using rbf_feature = feature::gaussian_rbf<double, 10>;
      static_assert(concepts::feature<rbf_feature, double, 11>);

      using linear_function = decltype(function::make_linear(std::declval<rbf_feature>()));
      static_assert(concepts::linear_function<linear_function, rbf_feature, double, 11>);			
      

    }
  }
}
