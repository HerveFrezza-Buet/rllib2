# RLlib2

The rllib2 is a C++ library for generic programming of reinforcement learning. In this repository, you will find the core matter, rllib2, and a rllib2-eigen library.

The former contains all the concepts and the main algorithms that can be written regardless the specific numerical library that is used for inner computations. The latter, rllib2-eigen, uses the eigen library to implement these computations. Further implementations with other numerical libraries, as gsl, may be considered in the future.

## Concepts

### MDPs

In RL, the goal is to apply a policy in order to drive a dynamical system (modelled as a MDP or a POMDP). We rely on gdyn for that purpose.

The reward is the report for transitions in this case, so the `mdp` concept in the rllib is a only a specification of the `system` concept of gdyn, where reports are double values. The "commands" are rather called "actions". We usually name `A`, the command type, and `S` the state type, which is also the observation type when we do not address POMDP issues. The dynamical system is sometimes referred to as "environment" in RL.

The `orbit_point` are still available, but the RL framework rather considers "transitions", called `sarsa`, which are slightly different. Running orbits in the rllib is thus the following:

```cpp
auto environment = make_some_dynamical_system();

A policy(const S& state) {..., return some_action;}

environment = initial_state;

for(auto [s, a, r, ss, aa]
    : gdyn::views::controller(environment, policy) // Provides successive actions
    | gdyn::views::orbit(environment)              // Get the orbit points...
    | rl2::views::sarsa) {                         // ... and collect (s, a, r, ss, aa) transitons from each one.
    // Nota : aa is an std::optional<A>, since there is no next action in case of reaching a terminal state.
}
```
