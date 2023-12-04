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

// not useful right now, but to answer problem described at line 272
// a convertor with Final State
struct S_convertor_FS : public S_convertor {

  // index of final state is maximal index + 1
  static std::size_t fs_index = S_convertor::from(gdyn::problem::cartpole::state{std::get<1>(S_convertor::limits[0]), std::get<1>(S_convertor::limits[1]), std::get<1>(S_convertor::limits[2]), std::get<1>(S_convertor::limits[3])});

  static gdyn::problem::cartpole::state to(std::size_t index)
  {
    // final state => NaN
    if (index == fs_index) {
      return gdyn::problem::cartpole::state{std::nan, std::nan, std::nan, std::nan};
    }
    return S_convertor::from(index);
  }

  static std::size_t from(const gdyn::problem::cartpole::state& s)
  {
    // if out of limits, final state
    if ((s.x < std::get<0>(S_convertor::limits[0])) or (s.x > std::get<1>(S_convertor::limits[0]))
        or (s.x_dot < std::get<0>(S_convertor::limits[1])) or (s.x_dot > std::get<1>(S_convertor::limits[1]))
        or (s.theta < std::get<0>(S_convertor::limits[2])) or (s.theta > std::get<1>(S_convertor::limits[2]))
        or (s.theta_dot < std::get<0>(S_convertor::limits[3])) or (s.theta_dot > std::get<1>(S_convertor::limits[3]))) {
      return fs_index;
    }
    return S_convertor(s);
  }
}: // S_convertor_FS

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


// Enumerable S and A for MDP
using S  = rl2::enumerable::count<gdyn::problem::cartpole::state,     S_convertor::size, S_convertor>;
using A  = rl2::enumerable::count<gdyn::problem::cartpole::direction, A_convertor::size, A_convertor>;
using SA = rl2::enumerable::pair<S, A>;

void test_convertor()
{
  // need an environment
  auto sys = gdyn::problem::cartpole::make();

  // teste conversion continuous - discrete
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
    auto no = *sys;
    auto d_no = static_cast<discrete_cartpole::observation_type::base_type>(*dsys);
    auto d_no_index = static_cast<std::size_t>(*dsys);
    bool nalive = sys;
    if (not alive) {
      std::cout << "**WARN** : transition *FROM* a terminal state" << std::endl;
    }
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << o << " (" << alive << ")"
              << " + a=" << cmd
              << " => no=" << no << " (" << nalive << ")"
              << " r=" << r
              << std::endl;
    std::cout << "    d_o_index=" << d_o_index
              << " d_o=" << d_o
              << " => d_no_index=" << d_no_index
              << " d_no=" << d_no
              << std::endl;
    if (not nalive) {
      std::cout << "  +--> reached a Terminal state" << std::endl;
    }
    // specific to cartpole : stop the loop if reward is zero
    if (r == 0) {
      std::cout << "  The transition was needed to get the 0 reward" << std::endl;
      break;
    }
  }

  /* **TODO********************
  Pose problème car ce qui est considéré comme l'état final devrait avoir
            une valeur de 0, mais ici il est possible que sa valeur change car il est
            confondu avec un état valide qui a pu déjà être modifié.
            << std::endl;
  Question : rl2::range::controller ou gdyn::views::orbit ou
            gdyn::views::sarsa tiennent compte du statut (boolean) du système ?
            << std::endl;
  - rl2::range::controller => non
   - gdyn::ranges::tick => non
   - gdyn::iterators::orbit => OUI
  */

  std::cout << std::endl;
  std::cout << "** Essayons en continu" << std::endl;
  // system initialization
  sys = continuous_cartpole::state_type(0,0,0,0);
  step = 0;
  for(auto [o, a, r, no, na]
        // TODO dire/expliquer que le rl2::ranges ne marche que sur les MDP
        : gdyn::ranges::controller(sys, rnd_policy)
        | gdyn::views::orbit(sys)
        | rl2::views::sarsa
        | std::views::take(50)) {
    bool nalive = sys;
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << o
              << " + a=" << a
              << " => no=" << no << " (" << nalive << ")"
              << " r=" << r
              << std::endl;
  }

  std::cout << std::endl;
  std::cout << "** Essayons en discret" << std::endl;
  // system initialization
  sys = continuous_cartpole::state_type(0,0,0,0);
  step = 0;
  for(auto [o, a, r, no, na]
        // TODO dire/expliquer que le rl2::ranges ne marche que sur les MDP
        : gdyn::ranges::controller(dsys, rnd_policy)
        | gdyn::views::orbit(dsys)
        | rl2::views::sarsa
        | std::views::take(50)) {
    auto d_o = static_cast<discrete_cartpole::observation_type::base_type>(o);
    auto d_o_index = static_cast<std::size_t>(o);
    auto d_no = static_cast<discrete_cartpole::observation_type::base_type>(o);
    auto d_no_index = static_cast<std::size_t>(o);
    bool nalive = sys;
    std::cout << std::boolalpha
              << "[" << step++ << "]"
              << " o=" << d_o_index << " " << d_o
              << " + a=" << static_cast<discrete_cartpole::command_type::base_type>(a)
              << " => no=" << d_no_index << " " << d_no << " (" << nalive << ")"
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
    std::size_t nb_epochs     = 2;
    std::size_t epoch_length  =  100;
  };
  Params learn_params;

  // gdyn::problem::cartpole is NOT a rl2::concepts::mdp
  static_assert(not rl2::concepts::mdp<continuous_cartpole>);
  // because it is NOT a gdyn::concept::transparent_system
  static_assert(not gdyn::concepts::transparent_system<continuous_cartpole>);
  // whereas is has a proper reward report
  static_assert(std::same_as<continuous_cartpole::report_type,double>);

  // TODO do we need transparent_system for MDP ??
  // => algo can not naively be applied to POMDP

  struct transparent_continuous_cartpole : public continuous_cartpole {
    using state_type       = continuous_cartpole::state_type;
    using command_type     = continuous_cartpole::command_type;
    using observation_type = continuous_cartpole::observation_type;
    using report_type      = continuous_cartpole::report_type;

    transparent_continuous_cartpole(const gdyn::problem::cartpole::parameters& param)
      : continuous_cartpole(param) {}

    void operator=(state_type state_init) {this->continuous_cartpole::operator=(state_init);}
    state_type state() const {return this->continuous_cartpole::operator*();}
  };

  // transparent_continuous_cartpole IS a rl2::concepts::mdp
  static_assert(rl2::concepts::mdp<transparent_continuous_cartpole>);


  // Pour utiliser rl2::critic::td::update il faut une fonction Q tabular
  // donc, passer par un transparent_dicrete_carpole.
  using transparent_discrete_cartpole = rl2::enumerable::system<S, S, A, transparent_continuous_cartpole>;
  gdyn::problem::cartpole::parameters sys_param;
  auto cont_mdp = transparent_continuous_cartpole(sys_param);
  auto discrete_mdp = transparent_discrete_cartpole(cont_mdp);
  static_assert(rl2::concepts::mdp<decltype(discrete_mdp)>);

  // Then a tabular Q
  std::array<double, SA::size> values;
  auto Q = rl2::tabular::make_two_args_function<S, A>(values.begin());

  // And policies
  auto greedy_policy         = rl2::discrete::greedy_ify(Q);
  auto epsilon_greedy_policy = rl2::discrete::epsilon_ify(greedy_policy, learn_params.epsilon, gen);

  auto random_state_generator = rl2::discrete::uniform_sampler<S>(gen);
  for(unsigned int epoch=0; epoch < learn_params.nb_epochs; ++epoch) {
    discrete_mdp = random_state_generator(); // We implement exploring starts.
    for(auto transition
	  : rl2::ranges::controller(discrete_mdp, epsilon_greedy_policy)
	  | gdyn::views::orbit(discrete_mdp)
	  | rl2::views::sarsa
	  | std::views::take(learn_params.epoch_length)) {
      // TODO sarsa td::evaluation_error vs QL td::discrete::optimal_error
      // TODO bellman_op et bellman_eval_op

      // Q-Learning
      double td_error = rl2::critic::td::discrete::optimal_error(Q, learn_params.gamma, transition);
      if (verbose) {
        std::cout << "update s:" << static_cast<std::size_t>(transition.s)
                  << " a:" << static_cast<std::size_t>(transition.a)
                  << " r=" << transition.r
                  << " td=" << td_error
                  << " oldQ=" << Q(transition.s, transition.a);
      }
      // TODO could return DeltaQ(s,a)
      rl2::critic::td::update(Q, transition.s, transition.a, learn_params.learning_rate, td_error);

      if (verbose) {
        std::cout << " -> newQ=" << Q(transition.s, transition.a)<< std::endl;
      }
    }
    // TODO change alpha
    // TODO change epsilon (useful for SARSA)
  }
  

  /*
  auto T = [sys = gdyn::problem::cartpole::make()] (const S& s, const A& a) mutable -> S {
    sys = static_cast<S::base_type>(s); // We init sys with s
    sys(static_cast<A::base_type>(a));  // We ask the continuous system to perform a transition.
    return *sys;                        // implicit conversion from gdyn::problem::cartpole::state to discrete S.
  };

  

  // - REWARD type with function r(s,a,s) -> double
  auto R = [] (const S&, const A&, const S&) -> double {
    // TODO difficult to write using env

    // assume that it is not called once terminated
    return 1.0;
  };

  // - TERMINAL type with function terminal(s) -> bool
  auto is_terminal = [sys] (const S& s) -> bool {
    // TODO difficult to write solely using env

    auto p = sys.param;
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
