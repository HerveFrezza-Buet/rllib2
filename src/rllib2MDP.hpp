#pragma once

#include <utility>
#include <functional>
#include <iterator>

#include<rllib2Specs.hpp>

namespace rl2 {

  template<typename STATE, typename ACTION>
  struct MDP {
  public:
    using state_type       = STATE;
    using observation_type = STATE; 
    using command_type     = ACTION;
    using report_type      = double;
    
    MDP()                      = delete;
    MDP(const MDP&)            = default;
    MDP(MDP&&)                 = default;
    MDP& operator=(const MDP&) = default;
    MDP& operator=(MDP&&)      = default;
    
  private:
    
    std::function<STATE  (const STATE&, const ACTION&)>               transition;
    std::function<double (const STATE&, const ACTION&, const STATE&)> reward;
    std::function<bool   (const STATE&)>                              terminal;
    
    state_type current_state;

  public:


    template<specs::transition<STATE, ACTION> TRANSITION, specs::reward<STATE, ACTION> REWARD, specs::terminal<STATE> TERMINAL>
    MDP(const TRANSITION& T, const REWARD& R, const TERMINAL& terminal)
      : transition(T), reward(R), terminal(terminal),
	current_state() {}

    void operator=(const state_type& s) {current_state = s;}
    observation_type operator*() const  {return current_state;}
    operator bool() const               {return !terminal(current_state);}
    
    report_type operator()(command_type command) {
      if(*this) {
	auto next_state = transition(current_state, command);
	double rew = reward(current_state, command, next_state);
	current_state = next_state;
	return rew;
      }
      return 0.; // Transition from a terminal state gives a 0 reward.
    }

    state_type state() const {return current_state;}
  };

  template<typename STATE, typename ACTION, specs::transition<STATE, ACTION> TRANSITION, specs::reward<STATE, ACTION> REWARD, specs::terminal<STATE> TERMINAL>
  auto make_mdp(const TRANSITION& T, const REWARD& R, const TERMINAL& terminal) {
    return MDP<STATE, ACTION>(T, R, terminal);
  }
}
