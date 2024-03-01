#pragma once

#include <random>
#include <gdyn.hpp>
#include <rllib2.hpp>


// This is the state space.
using S = gdyn::problem::cartpole::state;


// The action space is a enumerable version of gdyn::problem::cartpole::direction.
struct A_convertor {
  static constexpr std::size_t size() {return 2;} 
  static gdyn::problem::cartpole::direction to (std::size_t index)
  {
    switch(index) {
    case 0:
        return gdyn::problem::cartpole::direction::Left;
    default:
      return gdyn::problem::cartpole::direction::Right;
    }
  }
  static std::size_t from(const gdyn::problem::cartpole::direction& d)  {
    return static_cast<std::size_t>(d);
  }
};
using A = rl2::enumerable::set<gdyn::problem::cartpole::direction, A_convertor::size(), A_convertor>;


// Let us set up the Gaussian RBF for states.
// S is a struct with 4 double **contiguous** attributes.

struct wrapper {
  constexpr static std::size_t dim = 4; 
  S data;
  double* start;
  double* sentinel;
  wrapper(const S& data) : data(data) {
    start = reinterpret_cast<double*>(&(this->data));
    sentinel = start + dim;
  }
  const double* begin() const {return start;}
  const double* end()   const {return sentinel;}
};

using mu_type = rl2::nuplet::from<double, wrapper::dim>; 
using rbf = rl2::functional::gaussian<mu_type, S, wrapper>;

constexpr unsigned int nb_x_bins         = 5;
constexpr unsigned int nb_x_dot_bins     = 5;
constexpr unsigned int nb_theta_bins     = 5;
constexpr unsigned int nb_theta_dot_bins = 5;
using s_feature = rl2::features::rbfs<nb_x_bins * nb_x_dot_bins * nb_theta_bins * nb_theta_dot_bins,
					rbf>;  // This is our feature type, the appropriate number of basis functions.

auto make_bounds(double min, double max, unsigned int nb) {
  return std::make_tuple(min, max, .5*(max - min)/nb); // min, max, sigma
}


inline auto make_state_feature() {
  auto [x_min,         x_max,         x_sigma        ] = make_bounds(-4.8, 4.8, nb_x_bins);
  auto [x_dot_min,     x_dot_max,     x_dot_sigma    ] = make_bounds(-10., 10., nb_x_dot_bins);
  auto [theta_min,     theta_max,     theta_sigma    ] = make_bounds(-.21, .21, nb_theta_bins);
  auto [theta_dot_min, theta_dot_max, theta_dot_sigma] = make_bounds( -5.,  5., nb_theta_dot_bins);

  mu_type sigmas {x_sigma, x_dot_sigma, theta_sigma, theta_dot_sigma};
  auto gammas_ptr = std::make_shared<mu_type>(rl2::functional::gaussian_gammas_of_sigmas(sigmas));
  
  s_feature phi {};
  phi.rbfs = std::make_shared<s_feature::rbfs_type>();
  auto rbf_out = phi.rbfs->begin();
  
  mu_type mu;
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


// We will implement a linear parametrization of Q. We use eigen
// vectors to implement nuplets.
using params = rl2::eigen::nuplet::from<rl2::linear::discrete_a::q_dim_v<S, A, s_feature>>; 
using Q      = rl2::linear::discrete_a::q<params, S, A, s_feature>;                        




// Our cartpole needs to be redefined, in order to handle enumerable actions.
using cartpole = rl2::enumerable::discrete_a::system<S, S, A, gdyn::problem::cartpole::system>;


