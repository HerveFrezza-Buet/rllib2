/*
 Example en utilisant Cartpole en discrétisant les observations.
*/

#include <numbers>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>
#include <rllib2.hpp>




struct S_convertor {

  static constexpr std::size_t nb_bins {6};
  static constexpr std::size_t nb_dims {4};
  static constexpr std::size_t size    {nb_bins * nb_bins * nb_bins * nb_bins}; // nb_dims times.
  
  static constexpr std::array<std::tuple<double, double>, nb_dims> limits
    {{ {-4.8, 4.8},
       {-10.0, 10.0},
       {-12.0*std::numbers::pi/180.0, 12.0*std::numbers::pi/180.0},
       {-5.0, 5.0} }};

  
  static gdyn::problem::cartpole::state to(std::size_t index)
  {
    // indexes along each Obs dimensions
    std::array<std::size_t, nb_dims> indexes;

    for (int idi=indexes.size()-1; idi >= 0; --idi) {
      auto tmp_index = index % (nb_bins);
      index = (index - tmp_index) / (nb_bins);
      indexes[idi] = tmp_index;
    }

    std::size_t idi = 0;
    gdyn::problem::cartpole::state state;
    state.x = rl2::enumerable::utils::digitize::to_value(indexes[idi], std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    state.x_dot = rl2::enumerable::utils::digitize::to_value(indexes[idi], std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    state.theta = rl2::enumerable::utils::digitize::to_value(indexes[idi], std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    state.theta_dot = rl2::enumerable::utils::digitize::to_value(indexes[idi], std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);

    return state;
  }
  static std::size_t from(const gdyn::problem::cartpole::state& s)
  {
    // indexes along each Obs dimensions
    std::array<std::size_t,4> indexes;

    std::size_t idi = 0;
    indexes[idi] = rl2::enumerable::utils::digitize::to_index(s.x, std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    indexes[idi] = rl2::enumerable::utils::digitize::to_index(s.x_dot, std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    indexes[idi] = rl2::enumerable::utils::digitize::to_index(s.theta, std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);
    ++idi;
    indexes[idi] = rl2::enumerable::utils::digitize::to_index(s.theta_dot, std::get<0>(limits[idi]), std::get<1>(limits[idi]), nb_bins);

    // then convert array of indexes to one unique index
    std::size_t res = 0;
    for( auto index : indexes ) {
      res = res * (nb_bins) + index;
    }

    return res;
  }
}; // struct S_convertor

struct A_convertor {
  static constexpr std::size_t size {2}; 
  static gdyn::problem::cartpole::direction to (std::size_t index)
  {
    switch(index) {
    case 0:
        return gdyn::problem::cartpole::direction::Left;
    default:
      return gdyn::problem::cartpole::direction::Right;
    }
  }
  static std::size_t from(const gdyn::problem::cartpole::direction& d)
  {
    return static_cast<std::size_t>(d);
  }


}; // struct A_convertor

void test_convertor()
{
  // need an environment
  auto env = gdyn::problem::cartpole::make();
  //auto env = make_mdp()

  // teste conversion continuous - discrete
  // for the cartpole observation.
  {
    auto obs = *env;
    auto index = S_convertor::from(obs);
    auto obs_back = S_convertor::to(index);
    std::cout << "obs: " << obs << " (index = " << index << ")," << std::endl
	      << "     converted back to " << obs_back << '.' << std::endl;
  }
  
  {
    auto obs = S_convertor::to(0);
    std::cout << "obs: " << obs << " is got from 0 index." << std::endl;
  }

  {
    gdyn::problem::cartpole::system::observation_type obs_min, obs_max;
    unsigned int idd = 0;
    std::tie(obs_min.x,         obs_max.x)         = S_convertor::limits[idd++];
    std::tie(obs_min.x_dot,     obs_max.x_dot)     = S_convertor::limits[idd++];
    std::tie(obs_min.theta,     obs_max.theta)     = S_convertor::limits[idd++];
    std::tie(obs_min.theta_dot, obs_max.theta_dot) = S_convertor::limits[idd++];

    std::cout << "Min obs: " << obs_min << " (index = " << S_convertor::from(obs_min) << ")," << std::endl
	      << "Max obs: " << obs_max << " (index = " << S_convertor::from(obs_max) << ")." << std::endl;
  }

  // state for the first indexes
  std::cout << std::endl << std::endl;
  for (std::size_t id=0; id < 50; ++id)
    std::cout << std::setw(3) << id << ": " << S_convertor::to(id) << std::endl;
  std::cout << "..." << std::endl << std::endl;
}
template<typename RANDOM>
void test_transition(RANDOM& gen)
{

  // Enumerable S and A for MDP
  using S = rl2::enumerable::count<gdyn::problem::cartpole::state,     S_convertor::size, S_convertor>;
  using A = rl2::enumerable::count<gdyn::problem::cartpole::direction, A_convertor::size, A_convertor>;

  
    
  auto obs = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  auto act = gdyn::problem::cartpole::random_command(gen);

  {
    auto sys = gdyn::problem::cartpole::make();
    sys = obs;
    auto report = sys(act);
    auto next_obs = *sys;
    std::cout << "Transition from random state (continuous):"
	      << "  s : " << obs << std::endl
	      << "  a : " << act << std::endl
	      << "  r : " << report << std::endl
	      << "  s': " << next_obs << std::endl
	      << std::endl;
  }
  
  {
    // Let us build a transition function from the continuous
    // cartpole. This transition functions is the one for a discretized
    // cartpole.
    auto T = [sys = gdyn::problem::cartpole::make()] (const S& s, const A& a) mutable -> S {
      sys = static_cast<S::base_type>(s); // We init sys with s
      sys(static_cast<A::base_type>(a));  // We ask the continuous system to perform a transition.
      return *sys;                        // implicit conversion from gdyn::problem::cartpole::state to discrete S.
    };

    S discrete_s      {obs};
    S discrete_next_s {T(obs, act)}; // obs and act are implicitly converted to a S and A discrete instances.
    std::cout << "Transition from random state (discrete):"
	      << "  s             : " << obs << " (discretized as " << static_cast<gdyn::problem::cartpole::state>(discrete_s) << ", index = " << static_cast<std::size_t>(discrete_s) << ')' << std::endl
	      << "  a             : " << act << std::endl
	      << "  s' (discrete) : " << static_cast<gdyn::problem::cartpole::state>(discrete_next_s) << " (index = " << static_cast<std::size_t>(discrete_next_s) << ')'<< std::endl
	      << std::endl;
  }

}

template<typename RANDOM>
void test_mdp(RANDOM& gen) {
  /*
  auto sys = gdyn::problem::cartpole::make_environment();

  // Enumerable S and A for MDP
  using S = rl2::enumerable::count<gdyn::problem::cartpole::state,
                                   nb_bins^4,
                                   S_convertor>;
  using A = rl2::enumerable::count<gdyn::problem::cartpole::direction, 2, A_convertor>;
  using SA = rl2::enumerable::pair<S, A>;

  // to make an MDP we need
  // - TRANSITION type with function t(s,a) -> s
  auto T = [&env] (const S& s, const A& a) -> S {
    env = static_cast<S::base_type>(s);
    env(static_cast<A::base_type>(a));
    return S{*env};
  };

  // - REWARD type with function r(s,a,s) -> double
  auto R = [] (const S&, const A&, const S&) -> double {
    // TODO difficult to write using env

    // assume that it is not called once terminated
    return 1.0;
  };

  // - TERMINAL type with function terminal(s) -> bool
  auto is_terminal = [env] (const S& s) -> bool {
    // TODO difficult to write solely using env

    auto p = env.param;
    auto state = static_cast<S::base_type>(s);
    if (state.x < - p.x_threshold or state.x > p.x_threshold
        or state.theta < - p.theta_threshold_rad or state.theta > p.theta_threshold_rad) {
      return true;
    }
    return false;
  };

  // TODO would be nice to have make_mdp<S,A>(gdyn::system env)
  // or, at least, make_mdp<S,A>(gdyn::system env, R, is_terminal) if R (and is_terminal) are not in env
  // En utilisant cartpole, j'ai l'impression de définir les choses 2 fois.
  auto mdp = rl2::make_mdp<S, A>(T, R, is_terminal);

  // Then we can test it using a greedy and epsilon_greedy policies
  std::array<double, SA::size> qvalues;
  for(auto& value : qvalues) value = std::uniform_real_distribution(0., 1.)(gen);
  auto Q = rl2::tabular::make_two_args_function<S, A>(qvalues.begin());

  // Let us define a greedy policy.
  auto greedy_policy         = rl2::discrete::greedy_ify(Q);
  // auto epsilon_greedy_policy = rl2::discrete::epsilon_ify(greedy_policy, 0.2, gen);

  double total_gain = 0.0;
  for(auto [s, a, r, ss, aa]
        : rl2::ranges::controller(mdp, greedy_policy)
        | gdyn::views::orbit(mdp)
        | rl2::views::sarsa
        | std::views::take(10)) {
    std::cout << "s=" << static_cast<std::size_t>(s) << " + a=" << static_cast<std::size_t>(a);
    std::cout << " => r=" << r << " snext=" << static_cast<std::size_t>(ss) << std::endl;
    total_gain += r;
  }
  std::cout << "  Got a total gain of " << total_gain << std::endl;
  */
}


int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());


  if(argc != 2) {
    std::cout << "Usage: " << argv[0] << " [convertor | transition | mdp]" << std::endl;
    return 0;
  }

  std::string mode {argv[1]};

  if(mode == "convertor")  test_convertor();
  if(mode == "transition") test_transition(gen);
  if(mode == "mdp")        test_mdp(gen);
  return 0;

}
