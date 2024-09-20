#pragma once

#include <chrono>  // to time the various solvers
                   //
#include <rllib2.hpp>
#include <Eigen/Dense>
#include <utility>
#include <algorithm>

namespace rl2 {
  namespace eigen {
    namespace enumerable {
    namespace action {

      /* return an Eigen::Vector from
       * - phi(s), rl2::nuplet of double
       * - a, enumerable action
       *
       * returned vector is Zero everywhere, except between size(a)*N::s
       * dim and (size(a)+1)*N::dim where phi(s) is copied
       */
      template<rl2::concepts::nuplet N, rl2::concepts::enumerable::finite A>
      auto from(const N& ns, const A& a) {
	Eigen::Vector<double, N::dim*A::size()> vec;
	vec.setZero(); // eigen vectors are not initialized by the default constructor.

	int idv = static_cast<std::size_t>(a) * N::dim;

	std::ranges::copy(ns, vec.begin() + idv);
	return vec;
      }
    }
    }

    namespace critic {
      namespace enumerable {
      namespace action {

	template<bool compute_error,
		 rl2::concepts::enumerable::action::linear_qfunction Q,
		 rl2::concepts::policy<typename Q::state_type, typename Q::action_type> POLICY,
		 std::input_iterator TransitionIter, std::sentinel_for<TransitionIter> TransitionSentinel>
	requires rl2::concepts::sarsa<std::iter_value_t<TransitionIter>, typename Q::state_type, typename Q::action_type>
	double lstd(Q q, const POLICY& pi, double gamma, TransitionIter begin, TransitionSentinel end) {
	  Eigen::Matrix<double, Q::params_type::dim, Q::params_type::dim> sum_phiphi;
	  sum_phiphi.setZero();
	  Eigen::Vector<double, Q::params_type::dim> sum_phir;
	  sum_phir.setZero();


	  // LSTD-Q solves nw = argmin_w SUM( r_j+gamma*Phi_j+1.trans()*nw - Phi_j.trans()*w)
	  //      ie       SUM(Phi_j (Phi_j - gamma*Phi_j+1).trans()) * nw = SUM(Phi_j r_j)
	  // with j the indices in the replay buffer

	  // for_each_transition
	  for (auto it=begin; it!=end; ++it) {
	    // Build sum_phiphi from transitions, with Iter and Sentinel iterators
	    // reminder : std::vector<rl2::sarsa<S, A>> transitions;
	    // reminder : using Q = rl2::linear::enumerable::action::q<params, S, A, s_feature>;
	    //    *(Q.s_features)(t.s) - gamma * *(Q.s_features)(t.ss)
	    auto phi_s_eigen  = rl2::eigen::enumerable::action::from((*(q.s_feature))(it->s),  it->a);

	    auto phi_jgj = phi_s_eigen;
	    if (not it->is_terminal()) {
	      auto phi_ss_eigen = rl2::eigen::enumerable::action::from((*(q.s_feature))(it->ss), pi(it->ss));
	      phi_jgj = phi_jgj - gamma * phi_ss_eigen;
	    }
	    auto phiphi = phi_s_eigen * phi_jgj.transpose();

	    sum_phiphi += phiphi;

	    sum_phir += phi_s_eigen * it->r;
	  }

	  Eigen::Vector<double, Q::params_type::dim>& theta = *(q.params);
	  theta = sum_phiphi.completeOrthogonalDecomposition().solve(sum_phir);


	  if(compute_error)
	    return (sum_phiphi*theta - sum_phir).norm() / sum_phir.norm();
	  else
	    return 0;
	} // lstq()
      } // namespace action
      } // namespace enumerable
    } // namespace critic
  } // namespace eigen
} // namespace rl2
