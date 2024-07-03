#pragma once

#include <random>
#include <gdyn.hpp>
#include <rllib2.hpp>

#include "gdyn-system-mountain_car.hpp"


// This is the state space.
using S = gdyn::problem::mountain_car::state;

// The action space is a enumerable version of gdyn::problem::mountain_car::acceleration
struct A_convertor {
  // TODO pourquoi ce 3 il est pas dans la def du probleme ??
  static constexpr std::size_t size() {return 3;}
  static gdyn::problem::mountain_car::acceleration to(std::size_t index) {
    if(index ==  0)      return gdyn::problem::mountain_car::acceleration::Left;
    else if(index ==  1) return gdyn::problem::mountain_car::acceleration::None;
    else                 return gdyn::problem::mountain_car::acceleration::Right;
  }
  static std::size_t from(const gdyn::problem::mountain_car::acceleration& d)  {
    return static_cast<std::size_t>(d);
  }
};
using A = rl2::enumerable::set<gdyn::problem::mountain_car::acceleration, A_convertor::size(), A_convertor>;


// Let us set up the Gaussian RBF for states.  S is a struct with 4
// **contiguous** attributes with type double.

struct wrapper {
  constexpr static std::size_t dim = 2;
  const double* start;
  const double* sentinel;
  wrapper(const S& data)
    : start    (reinterpret_cast<const double*>(&data)),
      sentinel (reinterpret_cast<const double*>(&data) + dim) {}
  auto begin() const {return start;}
  auto end()   const {return sentinel;}
};

using mu_type = rl2::nuplet::from<double, wrapper::dim>; 
using rbf = rl2::functional::gaussian<mu_type, S, wrapper>;

constexpr unsigned int nb_pos_bins = 4;
constexpr unsigned int nb_vel_bins = 4;

// This is our feature type, with the appropriate number of basis functions.
using s_feature = rl2::features::rbfs<nb_pos_bins * nb_vel_bins,
                                      rbf>;

auto make_bounds(double min, double max, unsigned int nb) {
  return std::make_tuple(min, max, .5*(max - min)/nb); // min, max, sigma
}

inline auto make_state_feature() {
  auto [pos_min, pos_max, pos_sigma] = make_bounds(-1.2, 0.6, nb_pos_bins);
  auto [vel_min, vel_max, vel_sigma] = make_bounds(-0.07, 0.07, nb_vel_bins);

  mu_type sigmas {pos_sigma, vel_sigma};
  auto gammas_ptr = std::make_shared<mu_type>(rl2::functional::gaussian_gammas_of_sigmas(sigmas));
  
  s_feature phi {};
  phi.rbfs = std::make_shared<s_feature::rbfs_type>();
  auto rbf_out = phi.rbfs->begin();
  
  mu_type mu;
  for(unsigned int i = 0; i < nb_pos_bins; ++i) {
    mu[0] = rl2::enumerable::utils::digitize::to_value(i, pos_min, pos_max, nb_pos_bins);
    for(unsigned int j = 0; j < nb_vel_bins; ++j) {
      mu[1] = rl2::enumerable::utils::digitize::to_value(j, vel_min, vel_max, nb_vel_bins);
      *(rbf_out++) = {mu, gammas_ptr};
    }
  }

  return phi;
}

// TODO faire plus simple ? State -> Array -wrap-> nuplet --(lspi)-> Eigen::vec
// FEATURE doit avoir un operator()() -> nuplet et ::dim
// "Raw state" features are state.x, tanh(state.x_dot/10), state.theta, tanh(state.theta_dot/10)
// struct rawstate_wrapper {
//   // CONCEPTS: This is required by rl2::concepts::nuplet_wrapper.
//   constexpr static std::size_t dim = 4; // This tells how many scalars are used to represent the state

//   std::array<double, dim> data;         // This is a buffer for storing them when needed.

//   // CONCEPTS: This is required by rl2::concepts::nuplet_wrapper.
//   rawstate_wrapper(const S& state) {

//     auto it = data.begin();
//     *(it++) = state.x;
//     *(it++) = std::tanh( state.x_dot / 10.0 );
//     *(it++) = state.theta;
//     *(it++) = std::tanh( state.theta_dot / 10.0 );
//   }

//   // CONCEPTS: These are required by rl2::concepts::nuplet_wrapper.
//   auto begin() const {return data.begin();}
//   auto end()   const {return data.end();}

// }; // rawstate_wrapper

// TODO features needs a REF to array
struct raw_s_feature {
  constexpr static std::size_t dim = 6;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const S& state ) const
  {
    auto it = data->begin();
    *(it++) = 1.0;
    *(it++) = state.position;
    *(it++) = state.velocity;
    *(it++) = state.position * state.position;
    *(it++) = state.velocity * state.velocity;
    *(it++) = state.position * state.velocity;

    return rl2::nuplet::make_from_iterator<dim>(data->begin());
  }
}; // raw_s_feature


// We will implement a linear parametrization of Q. We use Eigen
// vectors to implement nuplets. The size required for the parameters
// is given at compiling time by rl2::linear::discrete_a::q_dim_v.
using params = rl2::eigen::nuplet::from<rl2::linear::discrete_a::q_dim_v<S, A, s_feature>>; 
using Q      = rl2::linear::discrete_a::q<params, S, A, s_feature>;                        

using raw_params = rl2::eigen::nuplet::from<rl2::linear::discrete_a::q_dim_v<S, A, raw_s_feature>>;
using raw_Q      = rl2::linear::discrete_a::q<raw_params, S, A, raw_s_feature>;

// Our mountain_car needs to be defined, in order to handle enumerable actions.
using mountain_car = rl2::enumerable::discrete_a::system<S, S, A, gdyn::problem::mountain_car::system>;
