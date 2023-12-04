/*
  Utiliser un gdyn::system comme satisfaisant un rl2::MDP
  (presque, à vérifier au niveau de transparent_system)

  Avec S contraint à être un VECTOR, peut utiliser une fonction
  Q(VECTOR S, Enumerable A) de type LINEAR_APPROX, avec poids w.

  - générer une trajectoire
  - mettre à jour les poids de Q avec LSTD-Q
*/

#include <chrono>
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include <list>
#include <random>
#include <rllib2.hpp>

// TODO global that could come from a gdyn::system description of the State space
constexpr std::size_t _state_dim {4};

// Convert cartpole::State to/from Eigen::VecXd
// in textbook, StateVec should be a column vector
using StateVec = Eigen::Vector<double, _state_dim>;

gdyn::problem::cartpole::state to(const StateVec& vec) {
  gdyn::problem::cartpole::state state;

  state.x = vec[0];
  state.x_dot = vec[1];
  state.theta = vec[2];
  state.theta_dot = vec[3];

  return state;
}

StateVec from(const gdyn::problem::cartpole::state& s) {
  StateVec vec {{s.x, s.x_dot, s.theta, s.theta_dot}};
  return vec;
}


std::ostream& operator<<(std::ostream& s, const StateVec& v)
{
  s << "( " << std::setprecision(3);
  for( auto val: v ) {
    s << val << ", ";
  }
  s << ")";

  return s;
}
// need also Enumerable Actions
struct A_convertor {
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

using S = StateVec;
using A = rl2::enumerable::count<gdyn::problem::cartpole::direction, 2, A_convertor>;

// For LSTD-Q, the "features" vector phi(s,a) are concatenation of |A|xStateVec
using PHI = Eigen::Vector<double, _state_dim*A::size>;
PHI make_feature(const gdyn::problem::cartpole::state& obs, const A& a) {
  PHI features;
  features.setZero();

  // add the state feature at index |S|*a_index position
  features.segment(static_cast<std::size_t>(a) * _state_dim, _state_dim) = from(obs);

  return features;
}
// and we need to store transitions of these features
// TODO could also use A=cartpole::Dir in fact, not leveraging that much on Enumrable
using PHI_SARSA = rl2::sarsa<PHI,A>;


// some helpers function for Eigen computation
template<typename M>
std::string shape (const M& mat)
{
  std::stringstream msg;
  msg << mat.rows() << "x" << mat.cols();

  return msg.str();
}

template<typename S>
struct QLINAPPROX {
  // In text book we should have that Q(s,a) = phi(s,a).transpose() * w
  // so w should be of length S::size * A::size
  // TODO optimizesd w as a matrix so that Q(s,a) = (S.transpose() * w)(a),
  //      i.e. S.transpose() * w is a row vector ?
  Eigen::Matrix<double, S::SizeAtCompileTime, A::size> W;

  // callable
  double operator()(const S& s, const A& a) {
    auto val = s.transpose() * W.col(static_cast<std::size_t>(a));
    return val;
  }

  void update_weight(const PHI& vec) {
    W = vec.reshaped(S::SizeAtCompileTime, A::size);
  }

  // dump
  std::string str_dump () const
  {
    std::stringstream dump;
    dump << "__Weights" << std::endl;
    for (int ida = 0; ida < W.cols(); ++ida) {
      auto dir = static_cast<gdyn::problem::cartpole::direction>(ida);
      dump << "  " << dir << ": " << W.col(ida).transpose() << std::endl;
    }

    return dump.str();
  }
}; // struct QLINAPPROX

// test converting to/from StateVec
void test_conversion()
{
  std::random_device rd;
  std::mt19937 gen(rd());

  auto env = gdyn::problem::cartpole::make();
  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  auto obs = *env;

  std::cout << "__Conversion *****************************************" << std::endl;
  std::cout << "  s=" << obs << std::endl;
  auto obs_vec = from(obs);
  std::cout << "  as vec=" << obs_vec << std::endl;

  auto obs_to = to(obs_vec);
  std::cout << "  back to " << obs_to << std::endl;
}

// test random init of Q, computation of Q(s,a)
void test_qval()
{
  std::random_device rd;
  std::mt19937 gen(rd());

  auto qapp = QLINAPPROX<S>();
  std::cout << "__Qval creation" << std::endl;
  std::cout << qapp.str_dump() << std::endl;

  std::cout << "__Qval random values" << std::endl;
  for( auto& val: qapp.W.reshaped()) {
    val = std::uniform_real_distribution<double>(0,1)(gen) * 2.0 - 1.0;
  }
  std::cout << qapp.str_dump() << std::endl;

  std::cout << "__Qval for a random state" << std::endl;
  auto env = gdyn::problem::cartpole::make();
  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  auto s = from(*env);
  auto a = A{0};
  auto qsa = qapp(s, a);
  std::cout << "  Q(s,a)=" << qsa << " for s=" << s;
  std::cout << " and " << static_cast<A::base_type>(a) << std::endl;
}

// Various ways to generate trajectories
void test_orbit()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  unsigned int step = 1;

  std::cout << "__Cartpole environment" << std::endl;
  auto env = gdyn::problem::cartpole::make();
  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());

  std::cout << "__Manual Orbit" << std::endl;
  for (auto command
         : gdyn::ranges::tick([&gen](){return gdyn::problem::cartpole::random_command(gen);})
         | std::views::take(10)) {
    auto s = from(*env);
    auto report = env(command);
    auto snext = from(*env);
    std::cout << "[" << step++ << "] ";
    std::cout << "s=" << s;
    std::cout << " x " << command;
    std::cout << " => " << report;
    std::cout << ", s=" << snext;
    std::cout << " (ended=" << std::boolalpha << not static_cast<bool>(env) <<")";
    std::cout << std::endl;
  }

  std::cout << "__gdyn Orbit" << std::endl;
  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  step = 1;
  for (auto [cur_obs, next_act_opt, prev_report_opt]
         : gdyn::ranges::tick([&gen](){return gdyn::problem::cartpole::random_command(gen);})
         | gdyn::views::orbit(env)
         | std::views::take(20)) {
    auto s = from(cur_obs);
    std::cout << "[" << step++ << "] ";
    std::cout << "at s=" << s;
    std::cout << "; r=";
    if (prev_report_opt) {std::cout << *prev_report_opt;} else {std::cout << "nil";}
    std::cout << " -> ";
    if (next_act_opt) {std::cout << *next_act_opt;} else {std::cout << "nil";}
    std::cout << std::endl;
  }

  std::cout << "__RL2 with SARSA view" << std::endl;
  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  step = 1;
  // s, a, r types are defined by env: cartpole::State, cartpole::Dire, double
  for (auto [s, a, r, snext, anext] // report is the reward here.
         : gdyn::ranges::tick([&gen](){return gdyn::problem::cartpole::random_command(gen);})
         | gdyn::views::orbit(env)
         | rl2::views::sarsa
         | std::views::take(20)) {
    std::cout << "[" << step++ << "] ";
    auto s_vec = from(s);
    std::cout << "s=" << s_vec;
    std::cout << " x " << a;
    std::cout << " => r=" << r;
    auto snext_vec = from(snext);
    std::cout << " s=" << snext_vec;
    std::cout << std::endl;
  }
}

void test_LSTDQ()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  double gamma {0.9};

  std::cout << "__LSTD-Q *********************************************" << std::endl;
  std::cout << "  create Cartpole environment" << std::endl;
  auto env = gdyn::problem::cartpole::make();

  std::cout << "  generate an orbit of features" << std::endl;
  std::list<PHI_SARSA> replay_buffer;

  env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());
  // see the first feature
  auto obs = *env;
  std::cout << "  start obs " << obs  << std::endl;
  auto feature = make_feature(*env, A{gdyn::problem::cartpole::direction::Right});
  std::cout << "  feature=" << feature.transpose() << std::endl;

  while (replay_buffer.size() < 2 * PHI::SizeAtCompileTime) {
    // reset environment
    env = gdyn::problem::cartpole::random_state(gen, gdyn::problem::cartpole::parameters());

    // s, a, r types are defined by env: cartpole::State, cartpole::Dir, double
    for (auto [s, a, r, snext, anext] // report is the reward here.
           : gdyn::ranges::tick([&gen](){return gdyn::problem::cartpole::random_command(gen);})
           | gdyn::views::orbit(env)
           | rl2::views::sarsa
           | std::views::take(20)) {
      if (anext) {
        replay_buffer.emplace_back( make_feature(s,a), A{a}, r,
                                    make_feature(snext, *anext) ); // no need for anext
      }
    };
    // repeat until we have at least weights.size() (or PHI::size) transitions
    std::cout << "  collected " << replay_buffer.size() << " samples" << std::endl;
  }

  // LSTD-Q solves nw = argmin_w SUM( r_j+gamma*Phi_j+1.trans()*nw - Phi_j.trans()*w)
  //      ie       SUM(Phi_j (Phi_j - gamma*Phi_j+1).trans()) * nw = SUM(Phi_j r_j)
  // with j the indices in the replay buffer
  std::cout << "  solving for new weights" << std::endl;
  Eigen::Matrix<double,
                PHI::SizeAtCompileTime,
                PHI::SizeAtCompileTime> sum_phiphi;
  sum_phiphi.setZero();
  // std::cout << "  sum_phiphi.shape=" << shape(sum_phiphi)<< std::endl;

  PHI sum_phir;
  sum_phir.setZero();

  for( const auto& t : replay_buffer ) {
    auto phi_jgj = (t.s - gamma * t.ss);
    // std::cout << "  phi_jgj.shape=" << shape(phi_jgj)<< std::endl;
    auto phiphi = t.s * phi_jgj.transpose();
    //std::cout << "  phiphi.shape=" << shape(phiphi)<< std::endl;

    sum_phiphi += phiphi;
    // std::cout << "  sum_phiphi.shape=" << shape(sum_phiphi)<< std::endl;

    sum_phir += t.s * t.r;
  }
  // Solving using Eigen ColPivHouseHolderQR decomposition
  // see https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
  auto t_start = std::chrono::high_resolution_clock::now();
  auto new_wvecQR = sum_phiphi.colPivHouseholderQr().solve(sum_phir);
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
  double rel_errorQR = (sum_phiphi*new_wvecQR - sum_phir).norm() / sum_phir.norm();
  std::cout << "  solution with colPivHouseholderQR()" << std::endl;
  std::cout << "  w=" << new_wvecQR.transpose() << std::endl;
  std::cout << "  error=" << rel_errorQR << std::endl;
  std::cout << "  in " << duration.count() << " ns" << std::endl;

  // Solving using Eigen leastsquare solution
  t_start = std::chrono::high_resolution_clock::now();
  auto new_wvecLS = sum_phiphi.completeOrthogonalDecomposition().solve(sum_phir);
  t_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start);
  double rel_errorLS = (sum_phiphi*new_wvecLS - sum_phir).norm() / sum_phir.norm();
  std::cout << "  solution with LeastSquare" << std::endl;
  std::cout << "  w=" << new_wvecLS.transpose() << std::endl;
  std::cout << "  error=" << rel_errorLS << std::endl;
  std::cout << "  in " << duration.count() << " ns" << std::endl;

  // use as the weighs for QLINAPP
  auto qapp = QLINAPPROX<S>();
  qapp.update_weight(new_wvecLS);
  std::cout << "  use as weights for Qval" << std::endl;
  std::cout << qapp.str_dump() << std::endl;
}

int main(int argc, char *argv[]) {

  // test_conversion();
  // test_qval();
  // test_orbit();
  test_LSTDQ();

  return 0;
}
