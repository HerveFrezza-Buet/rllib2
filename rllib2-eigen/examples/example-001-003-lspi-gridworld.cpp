
#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>
#include <rllib2-eigen.hpp>

std::random_device rd;
std::mt19937 gen(rd());

#define SIMU_W 5
#define SIMU_H 2
#define SIMU_GOAL 8

#define GAMMA 0.9
#define ALPHA_QLEARNING 1.0
#define MAX_EPOCH_LENGTH 30
#define BUFFER_SIZE 500
#define MAX_LSPI_ITER 30

// *****************************************************************************
// To use tabular QLearning, need Enumerable S and A
using g_system = gdyn::problem::grid_world::system<SIMU_W, SIMU_H, SIMU_GOAL>;

using enum_S = rl2::enumerable::set<typename g_system::state_type, 10>;

struct A_convertor {
  static g_system::command_type to (std::size_t index)
  {
    switch(index) {
    case  0: return gdyn::problem::grid_world::dir::North; break;
    case  1: return gdyn::problem::grid_world::dir::South; break;
    case  2: return gdyn::problem::grid_world::dir::West; break;
    default: return gdyn::problem::grid_world::dir::East; break;
    }
  }
  static std::size_t from (g_system::command_type a)
  {
    switch(a) {
    case  gdyn::problem::grid_world::dir::North: return 0; break;
    case  gdyn::problem::grid_world::dir::South: return 1; break;
    case  gdyn::problem::grid_world::dir::West: return 2; break;
    default: return 3; break;
    }
  }
}; // struct A_convertor
using enum_A = rl2::enumerable::set<typename g_system::command_type, 4, A_convertor>;

// using a enumerable system
using enum_system = rl2::enumerable::system<enum_S, enum_S, enum_A, g_system>;

// *****************************************************************************
enum_A optimal_policy(const g_system::state_type& s)
{
  switch(static_cast<int>(s)) {
  case 3:
    return enum_A{gdyn::problem::grid_world::dir::South};
    break;
  case 4:
  case 9:
    return enum_A{gdyn::problem::grid_world::dir::West};
    break;
  case 8:
    return enum_A{gdyn::problem::grid_world::dir::North};
    break;
  default:
    return enum_A{gdyn::problem::grid_world::dir::East};
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
  for( auto s: enum_S()) {
    std::cout << "Q( " << s << ", UDLR )=";
    for( auto a: enum_A() ) {
      std::cout << nice_double(q(s,a)) << "; ";
    }
    std::cout << std::endl;
  }
}

template<typename QTABLE>
std::string qval_in_line(const QTABLE& q, const enum_S& s)
{
  std::stringstream line;
  bool starting {true};
  for( auto a: enum_A() ) {
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
  for( auto s: enum_S()) {
    std::cout << "Q( " << s << ", UDLR )=";
    double best_qval = -1000000;
    auto best_a = enum_A{};
    for( auto a: enum_A() ) {
      auto val = q(s,a);
      std::cout << nice_double(val) << "; ";
      if (val > best_qval) {
        best_qval = val;
        best_a = a;
      }
    }
    std::cout << " ---> action: " << static_cast<g_system::command_type>(best_a);
    std::cout << " V(" << s << ")= " << best_qval << std::endl;
  }
}

// *****************************************************************************
// Features is a "one-hot" encoding of the state
struct s_features {
  constexpr static std::size_t dim = g_system::NB_STATES;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const g_system::state_type& state ) const
  {
    unsigned int id_state = static_cast<unsigned int>(state);

    auto it = data->begin();
    for (unsigned int i=0; i < g_system::NB_STATES; ++i) {
      if (i == id_state)
        *(it++) = 1.0;
      else
        *(it++) = 0.0;
    }

    return rl2::nuplet::make_from_iterator<dim>(data->begin());
  }
}; // struct s_features
struct enum_s_features {
  constexpr static std::size_t dim = g_system::NB_STATES;
  std::shared_ptr<std::array<double, dim>> data;

  auto operator()( const enum_S& state ) const
  {
    unsigned int id_state = static_cast<unsigned int>(state);

    auto it = data->begin();
    for (unsigned int i=0; i < g_system::NB_STATES; ++i) {
      if (i == id_state)
        *(it++) = 1.0;
      else
        *(it++) = 0.0;
    }

    return rl2::nuplet::make_from_iterator<dim>(data->begin());
  }
}; // struct enum_s_features
using enum_params = rl2::eigen::nuplet::from<rl2::linear::enumerable::action::q_dim_v<enum_S, enum_A,
										      enum_s_features>>;
using params = rl2::eigen::nuplet::from<rl2::linear::enumerable::action::q_dim_v<g_system::state_type,
										 enum_A,
										 s_features>>;
using QLinear = rl2::linear::enumerable::action::q<params, g_system::state_type, enum_A, s_features>;
using EnumQLinear = rl2::linear::enumerable::action::q<enum_params, enum_S, enum_A, enum_s_features>;

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
template<typename QTABLE>
void q_learning(QTABLE& q, enum_system& simulator, int nb_epoch)
{
  // using bellman optimality operator
  auto bellman_op = rl2::critic::td::enumerable::action::bellman::optimality<enum_S, enum_A, QTABLE>;

  for (int epoch=0; epoch < nb_epoch; ++epoch) {
    simulator = g_system::random_state(gen);

    for( auto transition
           : gdyn::views::pulse([](){return gdyn::problem::grid_world::random_command(gen);}) // random policy
           | gdyn::views::orbit(simulator)
           | rl2::views::sarsa
           | std::views::take(MAX_EPOCH_LENGTH)) {
      std::cout << ": s=" << static_cast<unsigned int>(transition.s) << " a=" << static_cast<int>(transition.a);
      std::cout << " --> [" << transition.r << "]";
      std::cout << " ns=" << static_cast<unsigned int>(transition.ss);
      if (transition.aa)
	std::cout << " na=" << static_cast<unsigned int>(transition.aa.value());
      else
	std::cout << " na=NULL";
      std::cout << " => Q(" << static_cast<unsigned int>(transition.s) << ", UDLR)=";
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
void fill(RANDOM_GENERATOR& gen, enum_system& simulator, const POLICY& policy,
          OutputIt out,
          unsigned int nb_samples, unsigned int max_episode_length) {
  unsigned int to_be_filled = nb_samples;
  while(to_be_filled > 0) {
    simulator = g_system::random_state(gen);
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
  g_system simulator;
  auto enum_simulator = enum_system(simulator);


  // FIRST : evaluate optimal Q/Policy using TABULAR QLearning
  // store Q in a Qtable
  std::array<double, g_system::NB_STATES * 4> table_values{}; // all elements = 0
  auto Q = rl2::enumerable::make_two_args_tabular<enum_S, enum_A>(table_values.begin());

  std::cout << "__Qval at Start" << std::endl;
  print_qval(Q);

  std::cout << "__Q-Learning for some steps" << std::endl;
  q_learning(Q, enum_simulator, 50);
  print_qval(Q);


  // SECONDLY, use LSTD-Q and LSPI with one-hot encoding as features
  std::cout << "__Parametrized Q using identity as Features" << std::endl;
  EnumQLinear q_lin{std::make_shared<enum_s_features>(), std::make_shared<enum_params>()};
  q_lin.s_feature->data = std::make_shared<std::array<double, s_features::dim>>();

  std::cout << ">> initial parameters" << std::endl;
  std::cout << "W=" << qlin_params_in_line(q_lin) << std::endl;

  // check encoder of s_features
  auto s = g_system::state_type{3};
  enum_s_features encoder{};
  encoder.data = std::make_shared<std::array<double, s_features::dim>>();
  auto features = encoder(s);
  std::cout << "s_features = ";
  for( auto v : features ) {
    std::cout << v << "/";
  }
  std::cout  << std::endl;

  // let's try to evaluate an optimal policy using LSTD-Q
  std::cout << "__estimated Value of optimal policy using linear approx" << std::endl;
  std::cout << "  filling buffer" << std::endl;
  std::vector<rl2::sarsa<enum_S, enum_A>> buffer;
  fill(gen, enum_simulator,
       [&enum_simulator](){return optimal_policy(*enum_simulator);},
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);
  std::cout << " got " << buffer.size() << " samples, ("
	    << std::ranges::count_if(buffer,
				     [](auto& buffer){return buffer.is_terminal();})
            << " are terminal transitions)." << std::endl;

  auto error = rl2::eigen::critic::enumerable::action::lstd<true>(q_lin,
								  optimal_policy,
								  GAMMA,
								  buffer.begin(), buffer.end());
  std::cout << " error = " << error << std::endl;
  std::cout << ">> parameters of the estimated optimal policy" << std::endl;
  std::cout << "W=" << qlin_params_in_line(q_lin) << std::endl;
  print_qval(q_lin);
  std::cout << ">>  as a reminder, the tabular policy of QLearning" << std::endl;
  print_qval_policy(Q);

  // THEN LSPI
  std::cout << "__And now with LSPI..." << std::endl;

  // QVal for temporary policy
  EnumQLinear q_lin_next{std::make_shared<enum_s_features>(), std::make_shared<enum_params>()};
  q_lin_next.s_feature->data = std::make_shared<std::array<double, s_features::dim>>();

  // init q_lin with random policy
  buffer.clear();
  fill(gen, enum_simulator,
       rl2::enumerable::uniform_sampler<enum_A>(gen),
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);
  rl2::eigen::critic::enumerable::action::lstd<false>(q_lin,
						      rl2::enumerable::action::random_policy<enum_S, enum_A>(gen),
						      GAMMA,
						      buffer.begin(), buffer.end());
  print_qval_policy(q_lin);

  double epsilon = 0.1;
  auto   greedy_on_q         = rl2::enumerable::greedy_ify(q_lin);
  auto   epsilon_greedy_on_q = rl2::enumerable::epsilon_ify(greedy_on_q, std::cref(epsilon), gen);
  std::cout << "Training using LSPI" << std::endl;
  for(unsigned int i = 0; i < MAX_LSPI_ITER; ++i) {
    // We re-sample the transition set, following the current q-values
    // with an epsilon-greedy policy.
    buffer.clear();
    fill(gen, enum_simulator,
         [&epsilon_greedy_on_q, &enum_simulator](){return epsilon_greedy_on_q(*enum_simulator);},
         std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);

    auto error = rl2::eigen::critic::enumerable::action::lstd<true>(q_lin_next,
								    greedy_on_q,
								    GAMMA,
								    buffer.begin(), buffer.end());
    std::cout << "  iteration " << std::setw(4) << i << " : error = " << error << std::endl;

    std::swap(q_lin, q_lin_next); // swapping q functions swaps their contents, which are share pointers.
    print_qval_policy(q_lin);
  }

  std::cout << ">> test the estimated policy" << std::endl;
  buffer.clear();
  fill(gen, enum_simulator,
       [&greedy_on_q, &enum_simulator](){return greedy_on_q(*enum_simulator);},
       std::back_inserter(buffer), BUFFER_SIZE, MAX_EPOCH_LENGTH);

  rl2::eigen::critic::enumerable::action::lstd<false>(q_lin_next,
						      greedy_on_q,
						      GAMMA,
						      buffer.begin(), buffer.end());

  print_qval_policy(q_lin_next);

  std::cout << ">> as a reminded, the QLearning tabular policy" << std::endl;
  print_qval_policy(Q);

  std::cout << "__END" << std::endl;
  return 0;
}
