#pragma once

// A very simple 2x5 grid world.
// The state is the number of the cell (0 to 9).
// The actions are U,D,L,R
// The reward is 1 in cell N°8

// This class fits the gdyn::concepts::system

#include <iostream>
#include <random>

#define GOAL_STATE 8
#define NB_STATE 10

struct gridworld {

  enum class dir : char {U='U', D='D', L='L', R='R'};

  // required by gdyn::concept::system
  using observation_type = int;
  using command_type = dir;
  using state_type = int;
  using report_type = double;

  template<typename RANDOM_GENERATOR>
  static state_type random_state(RANDOM_GENERATOR& gen) {
    return std::uniform_int_distribution<int>(0, NB_STATE)(gen);
  }

  template<typename RANDOM_GENERATOR>
  static dir random_command(RANDOM_GENERATOR& gen) {
    switch(std::uniform_int_distribution<int>(0, 3)(gen)) {
    case  0: return dir::U; break;
    case  1: return dir::D; break;
    case  2: return dir::L; break;
    default: return dir::R; break;
    }
  }

  state_type state {0};
  double reward {0.0};

  void compute_reward() {
    reward = 0.0;
    if (state == GOAL_STATE)
      reward = 1;
  }

  // This is required by the gdyn::specs::system concept.
  // This is for initializing the state of the system.
  gridworld& operator=(const state_type& init_state) {
    state = init_state;
    compute_reward();
    return *this;
  }

  // This is required by the gdyn::spec::system concept.
  // This returns the obsrvation corresponding to the system's state.
  observation_type operator*() const {
    return state;
  }

  // This is required by the gdyn::specs::system concept.
  // This it true if the system is not in a terminal state.
  operator bool() const {
    return state != GOAL_STATE;
  }

  // This is required by the gdyn::specs::system concept.
  // This performs a state transition.
  report_type operator()(command_type command)
  {
    switch (command) {
      case dir::U:
        if (state >= NB_STATE / 2)
          state -= NB_STATE / 2;
        break;
      case dir::D:
        if (state < NB_STATE / 2)
          state += NB_STATE / 2;
        break;
      case dir::L:
        if (state > NB_STATE / 2)
          state -= 1;
        else if (state > 0)
          state -= 1;
        break;
      case dir::R:
        if (state < NB_STATE / 2)
          state += 1;
        else if (state < NB_STATE)
          state += 1;
        break;
    }

    compute_reward();
    return reward;
  }
}; // struct gridworld

inline std::ostream& operator<<(std::ostream& os, gridworld::dir action) {
  return os << static_cast<char>(action);
}
