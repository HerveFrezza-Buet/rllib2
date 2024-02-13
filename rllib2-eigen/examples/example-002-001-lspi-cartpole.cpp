#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <tuple>

#include <Eigen/Dense>

#include <rllib2.hpp>
#include <rllib2-eigen.hpp>

#include <gdyn.hpp>


using S = Eigen::Vector<double, 4>;
using A = rl2::enumerable::count<gdyn::problem::cartpole::direction, 2>;


// We need to encapsulate the cartpole system into rllib2-eigen.
struct system {
private:
  static gdyn::problem::cartpole::state convert(S state) {return {state[0], state[1], state[2], state[3]};}
  static S convert(gdyn::problem::cartpole::state state) {return {state.x, state.x_dot, state.theta, state.theta_dot};}

public:
  
  
  using state_type = S;
  using observation_type = S;
  using command_type = A;
  using report_type = gdyn::problem::cartpole::system::report_type;

  gdyn::problem::cartpole::system borrowed_system;
  
  system(const gdyn::problem::cartpole::parameters& params): borrowed_system(params) {}

  system& operator=(const state_type& init_state) {borrowed_system = convert(init_state);}
  observation_type operator*() const {return convert(*borrowed_system);}
  operator bool() const {return borrowed_system;}
  report_type operator()(command_type command) {return borrowed_system(static_cast<gdyn::problem::cartpole::direction>(command));}
  
  template<typename RANDOM_GENERATOR>
  void random_init(RANDOM_GENERATOR& gen) {borrowed_system = gdyn::problem::cartpole::random_state(gen, borrowed_system.param);}
};


template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill(RANDOM_GENERATOR& gen, system& simulator, const POLICY& policy,
	  OutputIt out,
	  unsigned int nb_samples, unsigned int max_episode_length) {
  unsigned int to_be_filled = nb_samples;      
  while(to_be_filled > 0) {
    simulator.random_init(gen);
    std::ranges::copy(gdyn::views::pulse([&policy, &to_be_filled](){policy(); --to_be_filled;})
		      | gdyn::views::orbit(simulator)  
		      | rl2::views::sarsa
		      | std::views::take(to_be_filled)
		      | std::views::take(max_episode_length),
		      out);     
  }
}

auto make_bounds(double min, double max, unsigned int nb) {
  return std::make_tuple(min, max, .5*(max - min)/nb); // min, max, sigma
}

#define NB_TRANSITIONS     1000
#define MAX_EPOSODE_LENGTH   20

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // This will store transitions for LSTDQ computation.
  std::vector<rl2::sarsa<S, A>> transitions;

  // This is our feature space for the cartpole.
  
  constexpr unsigned int nb_bins = 5;
  auto [x_min,         x_max,         x_sigma        ] = make_bounds(0., 1., nb_bins);
  auto [x_dot_min,     x_dot_max,     x_dot_sigma    ] = make_bounds(0., 1., nb_bins);
  auto [theta_min,     theta_max,     theta_sigma    ] = make_bounds(0., 1., nb_bins);
  auto [theta_dot_min, theta_dot_max, theta_dot_sigma] = make_bounds(0., 1., nb_bins);
  
  rl2::eigen::feature::gaussian_rfb<S, nb_bins * nb_bins * nb_bins * nb_bins> s_phi {};
  auto rbf_out = std::back_inserter(s_phi.rbfs);
  
  for(unsigned int i = 0; i < nb_bins; ++i) {
    double x = rl2::utils::digitize::to_value(i, x_min, x_max, nb_bins);
    for(unsigned int j = 0; j < nb_bins; ++j) {
      double x_dot = rl2::utils::digitize::to_value(j, x_dot_min, x_dot_max, nb_bins);
      for(unsigned int k = 0; k < nb_bins; ++k) {
	double theta = rl2::utils::digitize::to_value(k, theta_min, theta_max, nb_bins);
	for(unsigned int l = 0; l < nb_bins; ++l) {
	  double theta_dot = rl2::utils::digitize::to_value(l, theta_dot_min, theta_dot_max, nb_bins);
	  *(rbf_out++) = {{x, x_dot, theta, theta_dot}, {x, x_dot_sigma, theta_sigma, theta_dot_sigma}};
	}
      }
    }
  }
  // ... ok, our feature object s_phi converting a state into a hight dimentional vector is set now.
  
  
  auto simulator = gdyn::problem::cartpole::make();

  // First, we fill the dataset with a random policy.
  std::cout << "Filling the dataset... " << std::flush;
  fill(gen, simulator,
       [&gen](){return gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters())},
       std::back_insterter(transitions),
       NB_TRANSITIONS, MAX_EPOSODE_LENGTH);

  // Let us iterate in order to apply lspi
  while(true) {
    auto q = rl2::eigen::critic::disrcete_a::lstd<A>(s_phi, transitions.begin(), transitions.end());
  }
       

  

  return 0;
}
