#pragma once

#include <chrono>  // to time the various solvers
                   //
#include <rllib2.hpp>
#include <Eigen/Dense>
#include <utility>
#include <algorithm>

namespace rl2 {
  namespace eigen {
    namespace discrete_a {

      /* return an Eigen::Vector from
       * - phi(s), rl2::nuplet of double
       * - a, enumerable action
       *
       * returned vector is Zero everywhere, except between size(a)*N::s
       * dim and (size(a)+1)*N::dim where phi(s) is copied
       */
      template<rl2::concepts::nuplet N, rl2::concepts::enumerable A>
      auto from(const N& ns, const A& a) {
	Eigen::Vector<double, N::dim*A::size()> vec;
	vec.setZero(); // eigen vectors are not initialized by the default constructor.

	int idv = static_cast<std::size_t>(a) * N::dim;

	std::ranges::copy(ns, vec.begin() + idv);
	return vec;
      }
    }

    namespace critic {
      namespace discrete_a {

	// // helper function for debug
	// template<typename M>
	// std::string shape (const M& mat)
	// {
	//   std::stringstream msg;
	//   msg << mat.rows() << "x" << mat.cols();

	//   return msg.str();
	// }

	template<bool compute_error,
		 rl2::concepts::discrete_a::linear_qfunction Q,
		 rl2::concepts::policy<typename Q::state_type, typename Q::action_type> POLICY,
		 std::input_iterator TransitionIter, std::sentinel_for<TransitionIter> TransitionSentinel>
	requires rl2::concepts::sarsa<std::iter_value_t<TransitionIter>, typename Q::state_type, typename Q::action_type>
	double lstd(Q q, const POLICY& pi, double gamma, TransitionIter begin, TransitionSentinel end) {
	  Eigen::Matrix<double, Q::params_type::dim, Q::params_type::dim> sum_phiphi;
	  sum_phiphi.setZero();
	  Eigen::Vector<double, Q::params_type::dim> sum_phir;
	  sum_phir.setZero();

	  // TODO print the params BEFORE
	  // std::cout << "__params BEFORE" << std::endl;
	  // std::cout << *(q.params) << std::endl;

	  // LSTD-Q solves nw = argmin_w SUM( r_j+gamma*Phi_j+1.trans()*nw - Phi_j.trans()*w)
	  //      ie       SUM(Phi_j (Phi_j - gamma*Phi_j+1).trans()) * nw = SUM(Phi_j r_j)
	  // with j the indices in the replay buffer

	  // for_each_transition
	  for (auto it=begin; it!=end; ++it) {
	    // Build sum_phiphi from transitions, with Iter and Sentinel iterators
	    // reminder : std::vector<rl2::sarsa<S, A>> transitions;
	    // reminder : using Q = rl2::linear::discrete_a::q<params, S, A,
	    // s_feature>;
	    //    *(Q.s_features)(t.s) - gamma * *(Q.s_features)(t.ss)
	    auto phi_s_eigen  = rl2::eigen::discrete_a::from((*(q.s_feature))(it->s),  it->a);
	    // std::cout << "**** Transition " << count_trans << std::endl;
	    // std::cout << " phi_s = " << phi_s_eigen << std::endl;
	    // TODO effacer si ca marche
	    // auto phi_ss_eigen = rl2::eigen::discrete_a::from((*(q.s_feature))(it->ss), pi(it->ss));
	    // auto phi_jgj = (phi_s_eigen - gamma * phi_ss_eigen);

			auto phi_jgj = phi_s_eigen;
			if (not it->is_terminal()) {
				auto phi_ss_eigen = rl2::eigen::discrete_a::from((*(q.s_feature))(it->ss), pi(it->ss));
				phi_jgj = phi_jgj - gamma * phi_ss_eigen;
			}
	    // std::cout << "  phi_jgj.shape=" << shape(phi_jgj)<< std::endl;
	    auto phiphi = phi_s_eigen * phi_jgj.transpose();
	    // std::cout << "  phiphi.shape=" << shape(phiphi)<< std::endl;

	    sum_phiphi += phiphi;

	    sum_phir += phi_s_eigen * it->r;
	  }
	  // std::cout << "  sum_phiphi.shape=" << shape(sum_phiphi)<< std::endl;

	  /*

	  // Solving using Eigen ColPivHouseHolderQR decomposition
	  // see https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
	  auto t_start = std::chrono::high_resolution_clock::now();
	  auto new_params_vecQR = sum_phiphi.colPivHouseholderQr().solve(sum_phir);
	  auto t_end = std::chrono::high_resolution_clock::now();
	  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
	  double rel_errorQR = (sum_phiphi*new_params_vecQR - sum_phir).norm() / sum_phir.norm();
	  std::cout << "  solution with colPivHouseholderQR()" << std::endl;
	  std::cout << "  params=" << new_params_vecQR.transpose() << std::endl;
	  std::cout << "  error=" << rel_errorQR << std::endl;
	  std::cout << "  in " << duration.count() << " ns" << std::endl;

	  // Solving using Eigen leastsquare solution
	  t_start = std::chrono::high_resolution_clock::now();
	  auto new_params_vecLS = sum_phiphi.completeOrthogonalDecomposition().solve(sum_phir);
	  t_end = std::chrono::high_resolution_clock::now();
	  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
	  double rel_errorLS = (sum_phiphi*new_params_vecLS - sum_phir).norm() / sum_phir.norm();
	  std::cout << "  solution with LeastSquare" << std::endl;
	  std::cout << "  params=" << new_params_vecLS.transpose() << std::endl;
	  std::cout << "  error=" << rel_errorLS << std::endl;
	  std::cout << "  in " << duration.count() << " ns" << std::endl;
	  
	  */

	  Eigen::Vector<double, Q::params_type::dim>& theta = *(q.params);
	  theta = sum_phiphi.completeOrthogonalDecomposition().solve(sum_phir);

	  if(compute_error)
	    return (sum_phiphi*theta - sum_phir).norm() / sum_phir.norm();
	  else
	    return 0;
	}
      }
    }
  }
}
