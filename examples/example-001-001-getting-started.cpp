#include <iostream>
#include <iomanip>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

/*

  Let us play the weakest link game (TV show), with a single player.
  At first, you are asked question A. If you answer correctly, you are
  asked question B, etc...

  If you give a wrong answer, you loose all the money in the pot.  If,
  before being asked question x, you say "bank !", you win the money
  in the pot, and the game restarts.

  When you provide a good answer to a question, some money (increasing
  with the question number) is put in the pot.

  We will model the answering as a probability to give the right
  answer, it is identical for all questions.

*/

// Let us define the state and action spaces (S_, A_) for this
using S_ = char; // This is the question tag (A, B, C, ...).
using A_ = bool; // true means "bank", false means "anwser".

// Indeed, here, we would like to iterate on state space and action
// space, since they are discrete sets. To do so, we can tweak S_ and
// A_ to provide them with iteration capabilities.

// For actions, the conversion is obvious, since in C++, bool is
// mapped naturally on to {0, 1}.
using A = rl2::enumerable<A_,  2>; // A::base_type is A_, i.e. bool.

// For characters, we need to map {'A', 'B', 'C', ... 'J'} onto {0, 1,
// ... 9}. This needs a shift (since A is the char value 65). To
// implement the conversion, we have to define a index convertion
// static functor.
struct S_index_convertor {
  static S_          to  (std::size_t index) {return static_cast<S_>(index + 65);         }
  static std::size_t from(S_          value) {return static_cast<std::size_t>(value) - 65;}
};
using S = rl2::enumerable<S_, 10, S_index_convertor>; // S::base_type is S_, i.e. char.

void show_index_conversion() {

  // Enumerable types have a base type, whose values are associated to
  // an index. Variables of enumarable types can be initialized and
  // affected with a base type value as well as an index value.
  
  S s1 {'C'};
  S s2 {std::size_t(5)};
  
  std::cout << "State " << static_cast<S::base_type>(s1) << " has index " << static_cast<std::size_t>(s1) << std::endl;
  std::cout << "State " << static_cast<S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;
  s2 = 'H';
  std::cout << "State " << static_cast<S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;
  s2 = std::size_t(0);
  std::cout << "State " << static_cast<S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;
  
  std::cout << std::endl;
}

void show_SA_enumeration() {

  // enumerable spaces can be iterated. The iterator 'it' can be
  // casted to a std::size_t value, providing us with the index of a
  // particular space value, and *it has the type base_type of the set
  // that has been intrumented to be enumerable.

  std::cout << "State space : " << S::size << " values" << std::endl;
  for(auto it = S::begin; it != S::end; ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;
  
  std::cout << "Action space : " << A::size << " values" << std::endl;
  for(auto it = A::begin; it != A::end; ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;
}

template <typename RANDOM_GENERATOR>
auto build_mdp(RANDOM_GENERATOR& gen, double correct_answer_probability) {
  
  // Let us define the reward table for the game. We can have arrays
  // since the state space size S::size is known at compiling time.
  std::array<double, S::size> rewards;
  auto it = rewards.begin();
  *(it++) = 0;
  *(it++) = 1;
  for(;it != rewards.end(); ++it) *it = *(it -1) * 2;

  // Let us display the reward table. The value of 'it' can be used
  // for accessing the elements of a tabular storing related to S
  // (here the rewards).
  std::cout  << "Rewards : " << std::endl;
  for(auto it = S::begin; it != S::end; ++it)
    std::cout << "  for state " << *it << " : " << rewards[it] << std::endl; 
  std::cout << std::endl;

  // Let us build a Markov Decision Process. It fits the
  // rl2::specs::MDP<S, A> concept, so it is a gdyn::specs::system
  // dynamical system.

  // We need a transition function. Let us use a lambda here.
  auto T = [&gen, p = correct_answer_probability](const S& s, const A& a) -> S {
    if(a) return 'A'; // If we bank (bank <=> true action), go to first question.
    if(std::bernoulli_distribution(p)(gen)) {             // If we answer correctly
      if(static_cast<S::base_type>(s) == 'J') return 'A'; // We go back to first question if we were at last question.
      return static_cast<std::size_t>(s) + 1;             // We go to next question otherwise.
    }
    return 'A'; // A bad answer leads us back to first question.
  };

  // This is the reward obtained for some s, a, s' transition. This is
  // a lambda function as well, which copies the reward table in its
  // lexical closure.
  auto R = [rewards](const S& s, const A& a, const S& ss) -> double {
    if(a) return rewards[static_cast<std::size_t>(s)]; // We get a reward if we bank.
    return 0;                                          // or 0 reward otherwise.
  };

  // In this problem, there are no terminal states.
  auto is_terminal = [](const S& s) {return false;}; // No state is terminal.

  // This builds a dynamical system.
  return rl2::make_mdp<S, A>(T, R, is_terminal);
}


int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.
  
  show_index_conversion();
  show_SA_enumeration();

  auto environment = build_mdp(gen, .75);
  environment = 'A';
  double total_gain = 0;
  for(auto [s, a, r, ss, aa]
	: gdyn::ranges::tick(rl2::uniform<A>(gen))
	| gdyn::views::orbit(environment)
	| rl2::views::sarsa
	| std::views::take(30)) {
    total_gain += r;
    std::cout << static_cast<S::base_type>(s) << " : ";
    if(a)
      std::cout << "bank  --> " << r << "$." << std::endl;
    else {
      std::cout << "answer... ";
      if(static_cast<S::base_type>(s) == 'J')       std::cout << "answer does not matters!";
      else if(static_cast<S::base_type>(ss) == 'A') std::cout << "bad answer.";
      else                                          std::cout << "correct!";
      std::cout << std::endl;
    }
  }
  std::cout << "--------------------" << std::endl
	    << "Total gain: " << total_gain << "$." << std::endl
	    << std::endl;
  
  return 0;
}
