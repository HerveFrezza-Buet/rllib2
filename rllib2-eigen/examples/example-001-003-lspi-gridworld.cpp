
#include <gdyn.hpp>
#include <rllib2.hpp>
#include "gdyn-system-gridworld.hpp"

std::random_device rd;
std::mt19937 gen(rd());

gridworld _simulator;
#define GAMMA 0.9
#define ALPHA_QLEARNING 0.1
#define MAX_EPOCH_LENGTH 30

// *****************************************************************************
// Check gridword : generate an orbit
void generate_orbit()
{

  _simulator = gridworld::random_state(gen)();
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
  auto bellman_op = rl2::critic::discrete::bellman::optimality<gridworld::state_type,
                                                               gridworld::command_type,
                                                               QTABLE>;

  for (unsigned int epoch=0; epoch < nb_epoch; ++epoch) {
    _simulator = gridworld::random_state(gen)();

    for( auto transition
           : gdyn::views::pulse([](){return gridworld::random_command(gen);}) // random policy
           | gdyn::views::orbit(_simulator)
           | rl2::views::sarsa
           | std::views::take(MAX_EPOCH_LENGTH)) {
      rl2::critic::td::update(q, transition.s, transition.a, ALPHA_QLEARNING,
                            rl2::critic::td::error(q, GAMMA, transition, bellman_op));
  }
}

// *****************************************************************************
int main(int argc, char *argv[])
{
  std::cout << "__generate ONE orbit, max length of 20" << std::endl;
  generate_orbit();


  std::cout << "__Q-Learning for some steps" << std::endl;
  // store Q in a Qtable
  std::array<double, NB_STATE> table_values;
  auto Q = rl2::tabular::make_two_args_function<gridworld::state_type, gridworld::command_type>(table_values.begin());

  q_learning(Q, 10);

  return 0;
}
