#pragma once

#include <rllib2.hpp>
#include <rllib2-eigen.hpp>
#include "rbf-utils.hpp"


namespace utils {
  namespace cartpole {

    constexpr unsigned int nb_x_bins         = 2;
    constexpr unsigned int nb_x_dot_bins     = 2;
    constexpr unsigned int nb_theta_bins     = 2;
    constexpr unsigned int nb_theta_dot_bins = 2;

    // This is our feature type, with the appropriate number of basis functions.
    using s_feature = rl2::features::rbfs<nb_x_bins * nb_x_dot_bins * nb_theta_bins * nb_theta_dot_bins,
					  rl2::problem::defs::cartpole::rbf>;

    inline auto make_state_feature() {
      auto [x_min,         x_max,         x_sigma        ] = rbf::make_bounds(-4.8, 4.8, nb_x_bins);
      auto [x_dot_min,     x_dot_max,     x_dot_sigma    ] = rbf::make_bounds(-10., 10., nb_x_dot_bins);
      auto [theta_min,     theta_max,     theta_sigma    ] = rbf::make_bounds(-.21, .21, nb_theta_bins);
      auto [theta_dot_min, theta_dot_max, theta_dot_sigma] = rbf::make_bounds( -5.,  5., nb_theta_dot_bins);

      rl2::problem::defs::cartpole::mu_type sigmas {x_sigma, x_dot_sigma, theta_sigma, theta_dot_sigma};
      auto gammas_ptr = std::make_shared<rl2::problem::defs::cartpole::mu_type>(
										rl2::functional::gaussian_gammas_of_sigmas(sigmas));
  
      s_feature phi {};
      phi.rbfs = std::make_shared<s_feature::rbfs_type>();
      auto rbf_out = phi.rbfs->begin();
  
      rl2::problem::defs::cartpole::mu_type mu;
      for(unsigned int i = 0; i < nb_x_bins; ++i) {
	mu[0] = rl2::enumerable::utils::digitize::to_value(i, x_min, x_max, nb_x_bins);
	for(unsigned int j = 0; j < nb_x_dot_bins; ++j) {
	  mu[1] = rl2::enumerable::utils::digitize::to_value(j, x_dot_min, x_dot_max, nb_x_dot_bins);
	  for(unsigned int k = 0; k < nb_theta_bins; ++k) {
	    mu[2] = rl2::enumerable::utils::digitize::to_value(k, theta_min, theta_max, nb_theta_bins);
	    for(unsigned int l = 0; l < nb_theta_dot_bins; ++l) {
	      mu[3] = rl2::enumerable::utils::digitize::to_value(l, theta_dot_min, theta_dot_max, nb_theta_dot_bins);
	      *(rbf_out++) = {mu, gammas_ptr};
	    }
	  }
	}
      }

      return phi;
    }

    // We will implement a linear parametrization of Q. We use Eigen
    // vectors to implement nuplets. The size required for the parameters
    // is given at compiling time by rl2::linear::enumerable::action::q_dim_v.
    using params = rl2::eigen::nuplet::from<rl2::linear::enumerable::action::q_dim_v<rl2::problem::defs::cartpole::S,
										     rl2::problem::defs::cartpole::A,
										     s_feature>>;
    using Q      = rl2::linear::enumerable::action::q<params,
						      rl2::problem::defs::cartpole::S,
						      rl2::problem::defs::cartpole::A,
						      s_feature>;

    // Our cartpole needs to be defined, in order to handle enumerable actions.
    using system = rl2::enumerable::action::system<rl2::problem::defs::cartpole::S,
						   rl2::problem::defs::cartpole::S,
						   rl2::problem::defs::cartpole::A,
						   gdyn::problem::cartpole::system>;

  } // namespace cartpole
} // namespace utils
