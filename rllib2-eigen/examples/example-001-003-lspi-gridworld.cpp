
#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <gdyn.hpp>
#include <random>
#include <rllib2.hpp>
#include <rllib2-eigen.hpp>
#include "gdyn-system-gridworld.hpp"

std::random_device rd;
std::mt19937 gen(rd());

gridworld _simulator;
#define GAMMA 0.9
#define ALPHA_QLEARNING 1.0
#define MAX_EPOCH_LENGTH 30
#define BUFFER_SIZE 500
#define MAX_LSPI_ITER 30

// *****************************************************************************
// To use tabular QLearning, need Enumerable S and A
using S = rl2::enumerable::set<typename gridworld::state_type, 10>;
struct A_convertor {
  static gridworld::command_type to (std::size_t index)
  {
    switch(index) {
    case  0: return gridworld::dir::U; break;
    case  1: return gridworld::dir::D; break;
    case  2: return gridworld::dir::L; break;
    default: return gridworld::dir::R; break;
    }
  }
  static std::size_t from (gridworld::command_type a)
  {
    switch(a) {
    case  gridworld::dir::U: return 0; break;
    case  gridworld::dir::D: return 1; break;
    case  gridworld::dir::L: return 2; break;
    default: return 3; break;
    }
  }
}; // struct A_convertor
using A = rl2::enumerable::set<typename gridworld::command_type, 4, A_convertor>;
// using a enumerable system
using enum_gridworld = rl2::enumerable::system<S, S, A, gridworld>;
auto enum_simulator = enum_gridworld(_simulator);

// *****************************************************************************
A optimal_policy(const gridworld::state_type& s)
{
  switch(static_cast<int>(s)) {
    case 3:
      return A{gridworld::dir::D};
      break;
    case 4:
    case 9:
      return A{gridworld::dir::L};
      break;
    case 8:
      return A{gridworld::dir::U};
      break;
    default:
      return A{gridworld::dir::R};
      break;
  }
}

// *****************************************************************************
inline std::string nice_double(double val)
{
  if (std::abs(val) < 0.0005)
    val = 0.0;

  std::stringstream nice;
  nice << std::setw(6) << std::setprecision(3) << val;

  return nice.str();
}
template<typename QTABLE>
void print_qval(const QTABLE& q)
{
  for( auto s: S()) {
    std::cout << "Q( " << s << ", UDLR )=";
    for( auto a: A() ) {
      std::cout << nice_double(q(s,a)) << "; ";
    }
    std::cout << std::endl;
  }
}
template<typename QTABLE>
std::string qval_in_line(const QTABLE& q, const S& s)
{
  std::stringstream line;
  bool starting {true};
  for( auto a: A() ) {
      if (starting) {
        starting = false;
      }
      else {
        std::cout << " / ";
      }
      std::cout << nice_double(q(s,a));
    }

  return line.str();
}
template<typename QTABLE>
void print_qval_policy(const QTABLE& q)
{
  for( auto s: S()) {
    std::cout << "Q( " << s << ", UDLR )=";
    double best_qval = -1000000;
    auto best_a = A{};
    for( auto a: A() ) {
      auto val = q(s,a);
      std::cout << nice_double(val) << "; ";
      if (val > best_qval) {
        best_qval = val;
        best_a = a;
      }
    }
    std::cout << " ---> action: " << static_cast<gridworld::command_type>(best_a);
    std::cout << " V(" << s << ")= " << best_qval << std::endl;
  }
}

// *****************************************************************************
// Features is a "one-hot" encoding of the state
struct s_features {
  constexpr static std::size_t dim = NB_STATE;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const gridworld::state_type& state ) const
  {
    int id_state = static_cast<int>(state);

    auto it = data->begin();
    for (int i=0; i < NB_STATE; ++i) {
      if (i == id_state)
        *(it++) = 1.0;
      else
        *(it++) = 0.0;
    }

    return rl2::nuplet::make_from_iterator<dim>(data->begin());
  }
}; // struct s_features
struct enum_s_features {
  constexpr static std::size_t dim = NB_STATE;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const S& state ) const
  {
    int id_state = static_cast<int>(state);

    auto it = data->begin();
    for (int i=0; i < NB_STATE; ++i) {
      if (i == id_state)
        *(it++) = 1.0;
      else
        *(it++) = 0.0;
    }

    return rl2::nuplet::make_from_iterator<dim>(data->begin());
  }
}; // struct enum_s_features
using enum_params = rl2::eigen::nuplet::from<rl2::linear::discrete_a::q_dim_v<S, A,
                                                                              enum_s_features>>;
using params = rl2::eigen::nuplet::from<rl2::linear::discrete_a::q_dim_v<gridworld::state_type,
                                                                         A,
                                                                         s_features>>;
using QLinear = rl2::linear::discrete_a::q<params, gridworld::state_type, A, s_features>;
using EnumQLinear = rl2::linear::discrete_a::q<enum_params, S, A, enum_s_features>;

template<typename QLINEAR>
std::string qlin_params_in_line(const QLINEAR& q_lin)
{
  std::stringstream line;
  bool starting {true};
  for (auto q_lin_param_it = q_lin.params->begin();
       q_lin_param_it != q_lin.params->end();
       ++q_lin_param_it) {
    if (starting) {
      starting = false;
    }
    else {
      std::cout << " / ";
    }
    std::cout << nice_double( *(q_lin_param_it) );
  }

  return line.str();
}
template<typename EIGENVEC>
std::string eigen_vec_in_line(const EIGENVEC& v)
{
  std::stringstream line;

  for (int i=0; i < v.size(); ++i) {
    if (i % 4 == 0) {
      line << "|| ";
    }
    line << v[i] << "; ";
  }
  return line.str();
}


// *****************************************************************************
// Check gridword : generate an orbit
void generate_orbit()
{
  _simulator = gridworld::random_state(gen);
  int step = 0;
  for(auto [s, a, r, ns, na] // report is the reward here.
        : gdyn::views::pulse([](){return gridworld::random_command(gen);}) // This is a source of random commands...
        | gdyn::views::orbit(_simulator)                                     // ... that feeds an orbit of the stystem...
        | rl2::views::sarsa
        | std::views::take(20)) {                                           // ... which will be interrupted after 20 steps at most.
    std::cout << step << ": s=" << s << " a=" << a;
    std::cout << " --> [" << r << "]";
    std::cout << " ns=" << ns;
    if (na)
      std::cout << " na=" << na.value();
    else
      std::cout << " na=NULL";
    //std::cout << " na=" << (na ? na.value() : "NULL");
    std::cout <<  std::endl;

    step++;
  }
}


template<typename QTABLE>
void q_learning(QTABLE& q, int nb_epoch)
{
  // using bellman optimality operator
  auto bellman_op = rl2::critic::td::discrete::bellman::optimality<S, A, QTABLE>;

  for (int epoch=0; epoch < nb_epoch; ++epoch) {
    _simulator = gridworld::random_state(gen);

    for( auto transition
           : gdyn::views::pulse([](){return gridworld::random_command(gen);}) // random policy
           | gdyn::views::orbit(enum_simulator)
           | rl2::views::sarsa
           | std::views::take(MAX_EPOCH_LENGTH)) {
    std::cout << ": s=" << static_cast<int>(transition.s) << " a=" << static_cast<int>(transition.a);
    std::cout << " --> [" << transition.r << "]";
    std::cout << " ns=" << static_cast<int>(transition.ss);
    if (transition.aa)
      std::cout << " na=" << static_cast<int>(transition.aa.value());
    else
      std::cout << " na=NULL";
    std::cout << " => Q(" << static_cast<int>(transition.s) << ", UDLR)=";
    std::cout << qval_in_line(q, transition.s);
    std::cout << std::endl;
    rl2::critic::td::update(q, transition.s, transition.a, ALPHA_QLEARNING,
                            rl2::critic::td::error(q, GAMMA, transition, bellman_op));
    }
  }
}

// *****************************************************************************
// This fills a transition dataset.
template<typename RANDOM_GENERATOR, typename POLICY, typename OutputIt>
void fill(RANDOM_GENERATOR& gen, enum_gridworld& simulator, const POLICY& policy,
	  OutputIt out,
	  unsigned int nb_samples, unsigned int max_episode_length) {
  unsigned int to_be_filled = nb_samples;
  while(to_be_filled > 0) {
    simulator = gridworld::random_state(gen);
    std::ranges::copy(gdyn::views::pulse(policy)
		      | gdyn::views::orbit(simulator)
		      | rl2::views::sarsa
		      | std::views::take(to_be_filled)
		      | std::views::take(max_episode_length)
		      | std::views::filter([&to_be_filled](const auto&){--to_be_filled; return true;}),
		      out);
  }
}

// *****************************************************************************
int main(int argc, char *argv[])
{
  std::cout << "__generate ONE orbit, max length of 20" << std::endl;
  generate_orbit();


  // store Q in a Qtable
  std::array<double, NB_STATE * 4> table_values{}; // all elements = 0
  auto Q = rl2::tabular::make_two_args_function<S, A>(table_values.begin());

  std::cout << "__Qval at Start" << std::endl;
  print_qval(Q);

  std::cout << "__Q-Learning for some steps" << std::endl;
  q_learning(Q, 50);
  print_qval(Q);

  std::cout << "__Parametrized Q using identity as Features" << std::endl;
  // QLinear q_lin{std::make_shared<s_features>(), std::make_shared<params>()};
  EnumQLinear q_lin{std::make_shared<enum_s_features>(), std::make_shared<enum_params>()};
  q_lin.s_feature->data = std::make_shared<std::array<double, s_features::dim>>();

  std::cout << ">> paramètres initiaux" << std::endl;
  std::cout << "W=" << qlin_params_in_line(q_lin) << std::endl;

  std::cout << ">> une transition 'à la main'" << std::endl;
  auto s = gridworld::state_type{3};
  _simulator = s;
  auto a = gridworld::command_type{gridworld::dir::D};
  auto r = _simulator(a);
  auto ns = *_simulator;
  auto na = gridworld::command_type{gridworld::dir::R};
  std::cout << "  3 x D => [" << r << "] " << ns << std::endl;

  // check s_features
  // s_features encoder{};
  enum_s_features encoder{};
  encoder.data = std::make_shared<std::array<double, s_features::dim>>();
  auto features = encoder(s);
  std::cout << "s_features = ";
  for( auto v : features ) {
    std::cout << v << "/";
  }
  std::cout  << std::endl;

  // for LSTQ step, do no forget to convert 'a' to its enumerable counterpart
  auto phi_s_eigen  = rl2::eigen::discrete_a::from((*(q_lin.s_feature))(S{s}), A{a});
  std::cout << "phi_s_eigen^T: " << eigen_vec_in_line(phi_s_eigen) << std::endl;

  auto phi_ss_eigen = rl2::eigen::discrete_a::from((*(q_lin.s_feature))(S{ns}), A{na});
  std::cout << "phi_ss_eigen^T: " << eigen_vec_in_line(phi_ss_eigen) << std::endl;

  auto phi_jgj = phi_s_eigen - GAMMA * phi_ss_eigen;
  std::cout << "phi_jgj^T: " << eigen_vec_in_line(phi_jgj) << std::endl;

  auto phiphi = phi_s_eigen * phi_jgj.transpose();
  std::cout << "phiphi:  " << phiphi << std::endl;

  // let's try to evaluate an optimal policy
  std::cout << "__estimated V of optimal policy using linear approx" << std::endl;
  std::cout << "  filling buffer" << std::endl;
  std::vector<rl2::sarsa<S, A>> buffer;
  fill(gen, enum_simulator,
       [](){return optimal_policy(*enum_simulator);},
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);
  std::cout << " got " << buffer.size() << " samples, ("
	    << std::ranges::count_if(buffer,
                               [](auto& buffer){return buffer.is_terminal();})
            << " are terminal transitions)." << std::endl;

  auto error = rl2::eigen::critic::discrete_a::lstd<true>(q_lin,
                                                          optimal_policy,
                                                          GAMMA,
                                                          buffer.begin(), buffer.end());
  std::cout << " error = " << error << std::endl;
  std::cout << ">> paramètres pour politique optimale" << std::endl;
  std::cout << "W=" << qlin_params_in_line(q_lin) << std::endl;
  print_qval(q_lin);
  std::cout << ">> pour rappel, QVal du QLearning" << std::endl;
  print_qval_policy(Q);

  std::cout << "__Et faisant du LSPI donc..." << std::endl;

  // QVal pour politique tampon
  EnumQLinear q_lin_next{std::make_shared<enum_s_features>(), std::make_shared<enum_params>()};
  q_lin_next.s_feature->data = std::make_shared<std::array<double, s_features::dim>>();

  // init q_lin with random policy
  buffer.clear();
  fill(gen, enum_simulator,
       rl2::discrete::uniform_sampler<A>(gen),
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);
  rl2::eigen::critic::discrete_a::lstd<false>(q_lin,
                                              rl2::discrete_a::random_policy<S, A>(gen),
                                              GAMMA,
                                              buffer.begin(), buffer.end());
  print_qval_policy(q_lin);

  double epsilon = 0.1;
  auto   greedy_on_q         = rl2::discrete::greedy_ify(q_lin);
  auto   epsilon_greedy_on_q = rl2::discrete::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 0; i < MAX_LSPI_ITER; ++i) {
    // We re-sample the transition set, following the current q-values
    // with an epsilon-greedy policy.
    buffer.clear();
    fill(gen, enum_simulator,
         [&epsilon_greedy_on_q](){return epsilon_greedy_on_q(*enum_simulator);},
         std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);

    auto error = rl2::eigen::critic::discrete_a::lstd<true>(q_lin_next,
							    epsilon_greedy_on_q,
							    GAMMA,
							    buffer.begin(), buffer.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    std::swap(q_lin, q_lin_next); // swapping q functions swaps their contents, which are share pointers.
    print_qval_policy(q_lin);
  }

  std::cout << ">> teste la politique obtenue" << std::endl;
  buffer.clear();
  fill(gen, enum_simulator,
       [&greedy_on_q](){return greedy_on_q(*enum_simulator);},
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);

  rl2::eigen::critic::discrete_a::lstd<false>(q_lin_next,
                                             greedy_on_q,
                                             GAMMA,
                                             buffer.begin(), buffer.end());

  print_qval_policy(q_lin_next);

  std::cout << ">> pour rappel, QVal du QLearning" << std::endl;
  print_qval_policy(Q);

  // std::cout << "__FittedQ iterations" << std::endl;
  // std::cout << "start with a random value function" << std::endl;
  // for( auto w: q_lin.s_feature->data) {
  //   w = std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0;
  // }
  // print_qval_policy(q_lin);

  std::cout << "__END" << std::endl;
  return 0;
}
