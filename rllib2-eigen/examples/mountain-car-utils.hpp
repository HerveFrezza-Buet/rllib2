#pragma once

#include <rllib2.hpp>
#include <rllib2-eigen.hpp>
#include "rbf-utils.hpp"

namespace utils {
  namespace mountain_car {

    constexpr unsigned int nb_pos_bins = 4;
    constexpr unsigned int nb_vel_bins = 4;

    // This is our feature type, with the appropriate number of basis functions.
    using s_feature = rl2::features::rbfs<nb_pos_bins * nb_vel_bins, rl2::problem::defs::mountain_car::rbf>;

    inline auto make_state_feature() {
      auto [pos_min, pos_max, pos_sigma] = rbf::make_bounds(-1.2, 0.6, nb_pos_bins);
      auto [vel_min, vel_max, vel_sigma] = rbf::make_bounds(-0.07, 0.07, nb_vel_bins);

      rl2::problem::defs::mountain_car::mu_type sigmas{pos_sigma, vel_sigma};
      auto gammas_ptr = std::make_shared<rl2::problem::defs::mountain_car::mu_type>(
										    rl2::functional::gaussian_gammas_of_sigmas(sigmas));

      s_feature phi{};
      phi.rbfs = std::make_shared<s_feature::rbfs_type>();
      auto rbf_out = phi.rbfs->begin();

      rl2::problem::defs::mountain_car::mu_type mu;
      for (unsigned int i = 0; i < nb_pos_bins; ++i) {
	mu[0] = rl2::enumerable::utils::digitize::to_value(i, pos_min, pos_max,
							   nb_pos_bins);
	for (unsigned int j = 0; j < nb_vel_bins; ++j) {
	  mu[1] = rl2::enumerable::utils::digitize::to_value(j, vel_min, vel_max,
							     nb_vel_bins);
	  *(rbf_out++) = {mu, gammas_ptr};
	}
      }

      return phi;
    }

    // We will implement a linear parametrization of Q. We use Eigen
    // vectors to implement nuplets. The size required for the parameters
    // is given at compiling time by rl2::linear::enumerable::action::q_dim_v.
    using params =
      rl2::eigen::nuplet::from<rl2::linear::enumerable::action::q_dim_v<rl2::problem::defs::mountain_car::S,
									rl2::problem::defs::mountain_car::A,
									s_feature>>;
    using Q = rl2::linear::enumerable::action::q<params,
						 rl2::problem::defs::mountain_car::S,
						 rl2::problem::defs::mountain_car::A,
						 s_feature>;

    // Our mountain_car needs to be defined, in order to handle enumerable actions.
    using system =
      rl2::enumerable::action::system<rl2::problem::defs::mountain_car::S,
				      rl2::problem::defs::mountain_car::S,
				      rl2::problem::defs::mountain_car::A,
				      gdyn::problem::mountain_car::system>;
  } // namespace mountain_car
} // namespace utils
