/*
 Example en utilisant Cartpole en discrétisant les observations.
*/

#include <numbers>
#include <random>
#include <string>
#include <rllib2.hpp>




struct S_convertor {

  constexpr std::size_t nb_bins {5};
  constexpr std::size_t nb_dims {4};
  constexpr std::size_t size_of() {
    std::size_t res = 1;
    for(int i = 0; i < nb_dims; ++i, res *= nb_bins);
    retirn res;
  }
  constexpr std::size_t size = size_of();
  
  constexpr std::array<std::tuple<double, double>, nb_dims> limits
    {{ {-4.8, 4.8},
       {-10.0, 10.0},
       {-12.0*std::numbers::pi/180.0, 12.0*std::numbers::pi/180.0},
       {-5.0, 5.0} }};

  
  static cartpole::State to(std::size_t index)
  {
    // indexes along each Obs dimensions
    std::array<std::size_t,4> indexes;

    for (int idi=indexes.size()-1; idi >= 0; --idi) {
      auto tmp_index = index % (_nb_bin_dim);
      index = (index - tmp_index) / (_nb_bin_dim);
      indexes[idi] = tmp_index;
    }

    std::size_t idi = 0;
    cartpole::State state;
    state.x = val_from_bin( indexes[idi], _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    state.x_dot = val_from_bin( indexes[idi], _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    state.theta = val_from_bin( indexes[idi], _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    state.theta_dot = val_from_bin( indexes[idi], _limits[idi][0], _limits[idi][1], _nb_bin_dim);

    return state;
  }
  static std::size_t from(const cartpole::State& s)
  {
    // indexes along each Obs dimensions
    std::array<std::size_t,4> indexes;

    std::size_t idi = 0;
    indexes[idi] = bin(s.x, _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    indexes[idi] = bin(s.x_dot, _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    indexes[idi] = bin(s.theta, _limits[idi][0], _limits[idi][1], _nb_bin_dim);
    ++idi;
    indexes[idi] = bin(s.theta_dot, _limits[idi][0], _limits[idi][1], _nb_bin_dim);

    // then convert array of indexes to one unique index
    std::size_t res = 0;
    for( auto index : indexes ) {
      res = res * (_nb_bin_dim) + index;
    }

    return res;
  }
}; // struct S_convertor

struct A_convertor {
  static cartpole::Dir to (std::size_t index)
  {
    switch(index) {
    case 0:
        return cartpole::Dir::L;
    default:
      return cartpole::Dir::R;
    }
  }
  static std::size_t from(const cartpole::Dir& d)
  {
    return static_cast<std::size_t>(d);
  }


}; // struct A_convertor

// TODO un exemple à part ?
void test_convertor()
{
  /*
  // need an environment
  auto env = cartpole::make_environment();
  //auto env = make_mdp()

  // teste conversion continuous - discrete
  // for the cartpole observation.
  auto obs = *env;
  cartpole::print_context("obs", obs, 0.0);

  auto index = S_convertor::from(obs);
  std::cout << "  => index=" << index << std::endl;

  auto from_index = S_convertor::to(index);
  cartpole::print_context("  converted back", from_index, 0.0);

  // State from the first index
  std::cout << "__convert from index=0" << std::endl;
  from_index = S_convertor::to(0);
  cartpole::print_context("  converted back", from_index, 0.0);

  // index of the limit inf of obs, should be 0
  unsigned int idp = 0;
  unsigned int idd = 0;
  obs.x = _limits[idd++][idp];
  obs.x_dot = _limits[idd++][idp];
  obs.theta = _limits[idd++][idp];
  obs.theta_dot = _limits[idd++][idp];
  std::cout << "__Limit inf" << std::endl;
  cartpole::print_context("limit inf", obs, 0.0);
  index = S_convertor::from(obs);
  std::cout << "  => index=" << index << std::endl;

  // index of the limit sup of obs, should be _nb_bin_dim^4-1 = 624
  idp = 1;
  idd = 0;
  obs.x = _limits[idd++][idp];
  obs.x_dot = _limits[idd++][idp];
  obs.theta = _limits[idd++][idp];
  obs.theta_dot = _limits[idd++][idp];
  std::cout << "__Limit sup" << std::endl;
  cartpole::print_context("limit sup", obs, 0.0);
  index = S_convertor::from(obs);
  std::cout << "  => index=" << index << std::endl;

  // State for the first indexes
  for (int id=0; id<static_cast<int>(2*_nb_bin_dim); ++id) {
    obs = S_convertor::to(id);
    std::cout << "__index=" << id << std::endl;
    cartpole::print_context("  =>", obs, 0.0);
  }
  */
}
template<typename RANDOM>
void test_transition(RANDOM gen)
{
  /*
  auto env = cartpole::make_environment();

  // Enumerable S and A for MDP
  using S = rl2::enumerable::count<cartpole::State,
                                   _nb_bin_dim^4,
                                   S_convertor>;
  using A = rl2::enumerable::count<cartpole::Dir, 2, A_convertor>;

  // to make an MDP we need
  // - TRANSITION type with function t(s,a) -> s
  auto T = [&env] (const S& s, const A& a) -> S {
    env = static_cast<S::base_type>(s);
    env(static_cast<A::base_type>(a));
    return S{*env};
  };

  // test transition from a random state
  env = cartpole::random_state(gen, env.param);
  auto obs = *env;
  auto act = cartpole::random_command(gen);
  auto report = env(act);
  auto next_obs = *env;
  print_context("starting from", obs, 0);
  std::cout << "we apply " << act << std::endl;
  print_context("to get", next_obs, report);

  // same with TRANSITION ?
  auto s = S{obs};
  auto a = A{act};
  auto s_next = T(s, a);
  std::cout << "Using TRANSITION" << std::endl;
  // TODO pk je peux pas directement écrire print_context("starting from", static_cast<S::base_type>(s), 0);
  auto s_base = static_cast<S::base_type>(s);
  print_context("starting from", s_base, 0);
  std::cout << "  with index=" << static_cast<std::size_t>(s) << std::endl;
  std::cout << "we apply " << static_cast<A::base_type>(a) << ", index=" << std::endl;
  auto s_next_base = static_cast<S::base_type>(s_next);
  print_context("to get", s_next_base, 0);
  std::cout << "  with index=" << static_cast<std::size_t>(s_next) << std::endl;

  */
}

template<typename RANDOM>
void test_mdp(RANDOM gen) {
  /*
  auto env = cartpole::make_environment();

  // Enumerable S and A for MDP
  using S = rl2::enumerable::count<cartpole::State,
                                   _nb_bin_dim^4,
                                   S_convertor>;
  using A = rl2::enumerable::count<cartpole::Dir, 2, A_convertor>;
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
