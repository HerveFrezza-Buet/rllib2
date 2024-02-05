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

  // The size of the discrete state is nb_bins^nb_dims, and we add 1
  // for out of limits. The max index is this out of limits state.
  static constexpr std::size_t size    {nb_bins * nb_bins * nb_bins * nb_bins + 1}; 
  
  static constexpr std::array<std::tuple<double, double>, nb_dims> limits
    {{ {-4.8, 4.8},
       {-10.0, 10.0},
       {-12.0*std::numbers::pi/180.0, 12.0*std::numbers::pi/180.0},
       {-5.0, 5.0} }};

  static constexpr gdyn::problem::cartpole::state out_of_limits_state {
    std::get<1>(limits[0]) + 1.,
    std::get<1>(limits[1]) + 1.,
    std::get<1>(limits[2]) + 1.,
    std::get<1>(limits[3]) + 1.};
  
  static gdyn::problem::cartpole::state to(std::size_t index)
  {
    if(index == size - 1)
      return out_of_limits_state;
    
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
    // if out of limits, final state
    if ((s.x < std::get<0>(limits[0])) or (s.x > std::get<1>(limits[0]))
        or (s.x_dot < std::get<0>(limits[1])) or (s.x_dot > std::get<1>(limits[1]))
        or (s.theta < std::get<0>(limits[2])) or (s.theta > std::get<1>(limits[2]))
        or (s.theta_dot < std::get<0>(limits[3])) or (s.theta_dot > std::get<1>(limits[3]))) {
      return size - 1;
    }
    
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
  static std::size_t from(const gdyn::problem::cartpole::direction& d)  {
    return static_cast<std::size_t>(d);
  }
}; // struct A_convertor


// Enumerable S and A for MDP
using S  = rl2::enumerable::set<gdyn::problem::cartpole::state,     S_convertor::size, S_convertor>;
using A  = rl2::enumerable::set<gdyn::problem::cartpole::direction, A_convertor::size, A_convertor>;
using SA = rl2::enumerable::pair<S, A>;

void test_convertor()
{
  // need an environment
  auto sys = gdyn::problem::cartpole::make();

  // test conversion continuous - discrete
  // for the cartpole observation.
  {
    auto obs = *sys;
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
  auto obs = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  auto act = gdyn::problem::cartpole::random_command(gen);

  {
    auto sys = gdyn::problem::cartpole::make();
    sys = obs;
    auto report = sys(act);
    auto next_obs = *sys;
    std::cout << "Transition from random state (continuous):" << std::endl
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
    std::cout << "Transition from random state (discrete):" << std::endl
	      << "  s             : " << obs << " (discretized as " << static_cast<gdyn::problem::cartpole::state>(discrete_s) << ", index = " << static_cast<std::size_t>(discrete_s) << ')' << std::endl
	      << "  a             : " << act << std::endl
	      << "  s' (discrete) : " << static_cast<gdyn::problem::cartpole::state>(discrete_next_s) << " (index = " << static_cast<std::size_t>(discrete_next_s) << ')'<< std::endl
	      << std::endl;
  }

}

template<typename RANDOM>
void test_discrete(RANDOM& gen) {
  std::cout << "__test_discrete" << std::endl;


  // need a system
  using continuous_cartpole = gdyn::problem::cartpole::system;
  auto sys = gdyn::problem::cartpole::make();
  // to be discretized
  using discrete_cartpole = rl2::enumerable::system<S, S, A, continuous_cartpole>;
  auto dsys = discrete_cartpole(sys);

  // system initialization
  sys = continuous_cartpole::state_type(0,0,0,0);

  // a random policy
  auto rnd_policy = [&gen](const continuous_cartpole::observation_type) {
    return gdyn::problem::cartpole::random_command(gen);
  };

  // display limites for x and theta
  std::cout << "*** Limits "
            << "x in [" << std::get<0>(S_convertor::limits[0])
            << ", " << std::get<1>(S_convertor::limits[0]) << "]"
            << " theta in [" << std::get<0>(S_convertor::limits[2])
            << ", " << std::get<1>(S_convertor::limits[2]) << "]"
            << std::endl;

  unsigned int step = 0;
  for(auto cmd
        : gdyn::ranges::controller(sys, rnd_policy)
        | std::views::take(50)) {
    auto o = *sys;
    auto d_o = static_cast<discrete_cartpole::observation_type::base_type>(*dsys);
    auto d_o_index = static_cast<std::size_t>(*dsys);
    bool alive = sys;
    auto r = sys(cmd);
    auto next_o = *sys;
    auto d_next_o = static_cast<discrete_cartpole::observation_type::base_type>(*dsys);
    auto d_next_o_index = static_cast<std::size_t>(*dsys);
    bool next_alive = sys;
    if (not alive) {
      std::cout << "**WARN** : transition *FROM* a terminal state" << std::endl;
    }
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << o << " (" << alive << ")"
              << " + a=" << cmd
              << " => next_o=" << next_o << " (" << next_alive << ")"
              << " r=" << r
              << std::endl;
    std::cout << "    d_o_index=" << d_o_index
              << " d_o=" << d_o
              << " => d_next_o_index=" << d_next_o_index
              << " d_next_o=" << d_next_o
              << std::endl;
    if (not next_alive) {
      std::cout << "  +--> reached a Terminal state" << std::endl;
    }
    // specific to cartpole : stop the loop if reward is zero
    if (r == 0) {
      std::cout << "  The transition was needed to get the 0 reward" << std::endl;
      break;
    }
  }

  std::cout << std::endl;
  std::cout << "** Let us try the continuous case" << std::endl;
  // system initialization
  sys = continuous_cartpole::state_type(0,0,0,0);
  step = 0;
  for(auto [o, a, r, next_o, next_a]
        : gdyn::ranges::controller(sys, rnd_policy)
        | gdyn::views::orbit(sys)
        | rl2::views::sarsa
        | std::views::take(50)) {
    bool next_alive = sys;
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << o
              << " + a=" << a
              << " => next_o=" << next_o << " (" << next_alive << ")"
              << " r=" << r
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << "** Let's experiment with the discrete case" << std::endl;
  // system initialization
  sys = continuous_cartpole::state_type(0,0,0,0);
  step = 0;
  for(auto [o, a, r, next_o, next_a]
        : gdyn::ranges::controller(dsys, rnd_policy)
        | gdyn::views::orbit(dsys)
        | rl2::views::sarsa
        | std::views::take(50)) {
    auto d_o = static_cast<discrete_cartpole::observation_type::base_type>(o);
    auto d_o_index = static_cast<std::size_t>(o);
    auto d_next_o = static_cast<discrete_cartpole::observation_type::base_type>(o);
    auto d_next_o_index = static_cast<std::size_t>(o);
    bool next_alive = sys;
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << d_o_index << " " << d_o
              << " + a=" << static_cast<discrete_cartpole::command_type::base_type>(a)
              << " => next_o=" << d_next_o_index << " " << d_next_o << " (" << next_alive << ")"
              << " r=" << r
              << std::endl;
  }
}

template<typename RANDOM>
void test_mdp(RANDOM& gen, bool verbose=false) {
  using continuous_cartpole = gdyn::problem::cartpole::system;

  std::cout << "__test_mdp **********************" << std::endl;
  struct Params {
    // Few parameters
    double learning_rate = .05;
    double gamma         = .95;
    double epsilon       = .20;

    // Experimental setup
    std::size_t nb_epochs     =  20;
    std::size_t epoch_length  = 100;
  };
  Params learn_params;

  // gdyn::problem::cartpole is a rl2::concepts::mdp
  static_assert(rl2::concepts::mdp<continuous_cartpole>);

  // For using rl2::critic::td::update we need a tabular Q function.
  // So we have to use a discrete cartpole problem.
  using discrete_cartpole = rl2::enumerable::system<S, S, A, continuous_cartpole>;

  
  gdyn::problem::cartpole::parameters sys_param;
  auto continuous_mdp = continuous_cartpole(sys_param);
  auto discrete_mdp = discrete_cartpole(continuous_mdp);
  static_assert(rl2::concepts::mdp<decltype(discrete_mdp)>);

  // Then a tabular Q
  std::array<double, SA::size> values;
  auto Q = rl2::tabular::make_two_args_function<S, A>(values.begin());

  // And policies
  auto greedy_policy         = rl2::discrete::greedy_ify(Q);
  auto epsilon_greedy_policy = rl2::discrete::epsilon_ify(greedy_policy, learn_params.epsilon, gen);

  auto random_state = [param = continuous_mdp.param, &gen]() {return gdyn::problem::cartpole::random_state(gen, param);};
    
  for(unsigned int epoch=0; epoch < learn_params.nb_epochs; ++epoch) {
   
    std::cout << "  epoch #" << epoch << std::endl;
    continuous_mdp = random_state(); // We implement exploring starts.
    for(auto transition
	  : gdyn::ranges::controller(discrete_mdp, epsilon_greedy_policy)
	  | gdyn::views::orbit(discrete_mdp)
	  | rl2::views::sarsa
	  | std::views::take(learn_params.epoch_length)) {

      // Q-Learning
      auto bellman_op = rl2::critic::td::discrete::bellman::optimality<S, A, decltype(Q)>;
      double td_error = rl2::critic::td::error(Q, learn_params.gamma, transition, bellman_op);
      if (verbose) {
        std::cout << "    update s:" << static_cast<std::size_t>(transition.s)
                  << " a:" << static_cast<std::size_t>(transition.a)
                  << " r=" << transition.r
                  << " td=" << td_error
                  << " oldQ=" << Q(transition.s, transition.a);
      }
      
      double deltaQ = rl2::critic::td::update(Q, transition.s, transition.a, learn_params.learning_rate, td_error);

      if (verbose) {
        std::cout << " -> newQ=" << Q(transition.s, transition.a) << std::endl;
      }
    }
  }
  
}


int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());


  if(argc != 2) {
    std::cout << "Usage: " << argv[0] << " [convertor | transition | discrete | mdp]" << std::endl;
    return 0;
  }

  std::string mode {argv[1]};

  if(mode == "convertor")  test_convertor();
  if(mode == "transition") test_transition(gen);
  if(mode == "discrete")   test_discrete(gen);
  if(mode == "mdp")        test_mdp(gen, true);
  return 0;

}
