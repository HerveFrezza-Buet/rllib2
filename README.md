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

### Enumerables

In RL, finite sets are often used for state type (in toy problems mainly), as well as for action type (very often). So the library offers a way to define finite types. The point is to associate all possible values with consecutive integers, starting from 0. A C++ type having these features fit the `enumarable` concept of rllib2.

The main issue with enumerable is the functions converting a value to its index, and vice-versa.

```cpp
// As bool naturally casts into std::size_t, this is straightforward.
using A = rl2::enumerable::set<bool, 2>; // A::base_type is bool, A::size() is 2.
 
// If we need to map {'A', 'B', 'C', ... 'J'} onto {0, 1,
// ... 9}, we have to implement a shift (since A is the char value 65). To
// implement the conversion, we have to define a index convertion
// static functor.
struct S_index_convertor {
  static S_to(std::size_t index)   {return static_cast<char>(index + 65); }
  static std::size_t from(S_value) {return static_cast<std::size_t>(value) - 65;}
};
using S = rl2::enumerable::set<char, 10, S_index_convertor>; // S::base_type is char, A::size() is 10
 

S state1 {'D'};                                       // The state D
std::size_t D_idf = static_cast<std::size_t>(state1); // D_idf = 3
S state2 {D_idf + 3};                                 // The state 3 steps after D_idf, i.e. G
char value = static_cast<char>(state2);               // value = 'G'
```

It also works for pairs:


```cpp
using SA = rl2::enumerable::pair<S, A>;

auto sa = std::make_pair('G', false);
SA state3 {sa};
std::size_t Gfalse_idf = static_cast<std::size_t>(sa); // Gfalse_idf = 12
SA state4 {std::size_t(15)};
auto s_a = static_cast<std::pair<char, bool>>(state4); // s_a = {'H', true}
```

In many algorithms, we need to build tables indexed by the discrete
values of S (or A). Using an integer starting from 0 enables to get
rid of `std::map`s for that purpose, since random access containers,
more efficient, can be used.


### Q functions

Rllib2, as expected, offers the concept `q_function`, which are `S, A -> double` functions. It is often convenient to enable partial application for Q-functions (typically when argmax_a Q(s, a) = argmax_a Q_s(a), Q_s being a partial application of Q). This is implemented as the `two_args_function` concept. If f fits that concept,  f is a `X, Y -> Z` functon, f(x) is a `Y -> Z` function.

We have `tabular::two_args_function` when both arguments are from an enumerable type, and `discrete_a::two_args_function` when only the second one is enumerable.


