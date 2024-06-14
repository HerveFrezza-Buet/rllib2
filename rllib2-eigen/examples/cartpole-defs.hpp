#pragma once

#include <random>
#include <gdyn.hpp>
#include <rllib2.hpp>


// This is the state space.
using S = gdyn::problem::cartpole::state;

// The action space is a enumerable version of gdyn::problem::cartpole::direction.
struct A_convertor {
  // TODO pourquoi ce 2 il est pas dans la def du probleme ??
  static constexpr std::size_t size() {return 2;} 
  static gdyn::problem::cartpole::direction to(std::size_t index) {
    if(index ==  0) return gdyn::problem::cartpole::direction::Left;
    else            return gdyn::problem::cartpole::direction::Right;
  }
  static std::size_t from(const gdyn::problem::cartpole::direction& d)  {
    return static_cast<std::size_t>(d);
  }
};
using A = rl2::enumerable::set<gdyn::problem::cartpole::direction, A_convertor::size(), A_convertor>;


// Let us set up the Gaussian RBF for states.  S is a struct with 4
// **contiguous** attributes with type double.

struct wrapper {
  constexpr static std::size_t dim = 4; 
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

constexpr unsigned int nb_x_bins         = 2;
constexpr unsigned int nb_x_dot_bins     = 2;
constexpr unsigned int nb_theta_bins     = 2;
constexpr unsigned int nb_theta_dot_bins = 2;

// This is our feature type, with the appropriate number of basis functions.
using s_feature = rl2::features::rbfs<nb_x_bins * nb_x_dot_bins * nb_theta_bins * nb_theta_dot_bins,
                                      rbf>;

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



// // TODO ce qui suit ne marche pas
// // required for the satisfaction of ‘feature<S_FEATURE, S>’ [with S_FEATURE = raw_s_feature; S = gdyn::problem::cartpole::state]
// // in requirements with ‘const FEATURE cf’, ‘const X cx’ [with X = gdyn::problem::cartpole::state; FEATURE = raw_s_feature]
// // note: the required expression ‘cf(cx)’ is invalid
// // {cf(cx)} -> std::ranges::input_range;
// // TODO si j'ajoute const à operator()
// // In member function ‘auto raw_s_feature::operator()(const S&) const’:
// // error: assignment of read-only location ‘*(it ++)’
// // *(it++) = state.x;

// TODO features needs a REF to array
struct raw_s_feature {
  constexpr static std::size_t dim = 4;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const S& state ) const
  {
    auto it = data->begin();
    *(it++) = state.x;
    *(it++) = std::tanh( state.x_dot / 10.0 );
    *(it++) = state.theta;
    *(it++) = std::tanh( state.theta_dot / 10.0 );

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

// Our cartpole needs to be defined, in order to handle enumerable actions.
using cartpole = rl2::enumerable::discrete_a::system<S, S, A, gdyn::problem::cartpole::system>;


