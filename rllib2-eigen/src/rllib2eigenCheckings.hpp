#pragma once

#include <tuple>

#include <rllib2.hpp>

#include <rllib2eigenFunctional.hpp>
#include <rllib2eigenDiscreteA.hpp>

namespace rl2 {
  namespace eigen {
    namespace checkings {

      using polynomial_feature = feature::polynomial<10>;
      static_assert(concepts::feature<polynomial_feature, double, 11>);

      using rbf_feature = feature::gaussian_rbf<double, 10>;
      static_assert(concepts::feature<rbf_feature, double, 11>);

      using linear_function = decltype(function::make_linear(std::declval<rbf_feature>()));
      static_assert(concepts::linear_function<linear_function, rbf_feature, double, 11>);			

      using S = double;
      using A = rl2::enumerable::set<char, 3>;
      using linear_q_discrete_a = feature::discrete_a::linear<S, A, 10, rbf_feature>;
      static_assert(concepts::linear_function<linear_q_discrete_a, feature::discrete_a::phi<S, A, 10, rbf_feature>, std::pair<S, A>, 3 * 10>);
    }
  }
}
