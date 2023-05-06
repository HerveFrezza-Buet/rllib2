#pragma once

#include <functional>

#include<rllib2Specs.hpp>

namespace rl2 {

  template<typename STATE, typename ACTION>
  struct MDP {
  public:
    using state_type       = STATE;
    using observation_type = std::tuple<STATE, double>; 
    using command_type     = ACTION;
    
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
    double last_reward;

  public:


    template<specs::transition<STATE, ACTION> TRANSITION, specs::reward<STATE, ACTION> REWARD, specs::terminal<STATE> TERMINAL>
    MDP(const TRANSITION& T, const REWARD& R, const TERMINAL& terminal)
      : transition(T), reward(R), terminal(terminal),
	current_state(), last_reward(0) {}

    void operator=(const state_type& s) {current_state = s;}
    observation_type operator*() const  {return {current_state, last_reward};}
    operator bool() const               {return !terminal(current_state);}
    
    void operator()(command_type command) {
      if(*this) {
	auto next_state = transition(current_state, command);
	last_reward = reward(current_state, command, next_state);
	current_state = next_state;
      }
    }
  };

  template<typename STATE, typename ACTION, specs::transition<STATE, ACTION> TRANSITION, specs::reward<STATE, ACTION> REWARD, specs::terminal<STATE> TERMINAL>
  auto make_mdp(const TRANSITION& T, const REWARD& R, const TERMINAL& terminal) {
    return MDP<STATE, ACTION>(T, R, terminal);
  }
}
