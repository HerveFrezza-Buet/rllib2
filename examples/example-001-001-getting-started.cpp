#include <iostream>
#include <iomanip>
#include <array>

#include <rllib2.hpp>

/*

  Let us play the weakest link game (TV show), with a single player.
  At first, you are asked question A. If you answer correctly, you are asked question B, etc...

  If you give a wrong answer, you loose all the money in the pot.
  If, before being asked question x, you say "bank !", you win the money in the pot, and the game restarts.

  When you provide a good answer to a question, some money (increasing with the question number) is put in the pot.

  We will model the answering as a probability to give the right answer, it is identical for all questions.

*/

// Let us define the state and action spaces (S_, A_) for this
using S_ = char; // This is the question tag (A, B, C, ...).
using A_ = bool; // true means "bank", false means "anwser".

// Indeed, here, we would like to iterate on state space and action
// space, since they are discrete sets. To do so, we can tweak S_ and
// A_ to provide them with iteration capabilities.

// For actions, the conversion is obvious, since in C++, bool is
// mapped naturally on to {0, 1}.
using A = rl2::enumerable<A_,  2>;

// For characters, we need to map {'A', 'B', 'C', ... 'J'} onto {0, 1,
// ... 9}. This needs a shift (since A is the char value 65). To
// implement the conversion, we have to define a index convertion
// static functor.
struct S_index_convertor {
  static S_          to  (std::size_t index) {return static_cast<S_>(index + 65);         }
  static std::size_t from(S_          value) {return static_cast<std::size_t>(value) - 65;}
};
using S = rl2::enumerable<S_, 10, S_index_convertor>;


int main(int argc, char* argv[]) {

  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  S some_state{'A'}
  std::cout << "State " << std::static_cast<' << " has index "

  // enumerable spaces can be iterated. The iterator it can be casted
  // to a std::size_t value, providing us with the index of a
  // particular space value, and *it has the type of the set that has
  // been intrumented to be enumerable.

  std::cout << "State space : " << S::size << " values" << std::endl;
  for(auto it = S::begin; it != S::end; ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;
  
  std::cout << "Action space : " << A::size << " values" << std::endl;
  for(auto it = A::begin; it != A::end; ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;


  // Let us define the reward scales for the game. We can have static
  // arrays since the state spaces size is known at compiling time.
  std::array<double, S::size> rewards;
  auto it = rewards.begin();
  *(it++) = 0;
  *(it++) = 1;
  for(;it != rewards.end(); ++it) *it = *(it -1) * 2;

  std::cout  << "Rewards : " << std::endl;
  for(auto it = S::begin; it != S::end; ++it)
    std::cout << "  for state " << *it << " : " << rewards[it] << std::endl; 
  std::cout << std::endl;
  
  
  return 0;
}
