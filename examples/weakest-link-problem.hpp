// This is the definition of a Markovian Decision Process modelling
// the weakest link game (with a single player).


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

namespace weakest_link {
  
  // Let us define the state and action spaces (S_, A_) for this
  using S_ = char; // This is the question tag (A, B, C, ...).
  using A_ = bool; // true means "bank", false means "anwser".

  // Indeed, here, we would like to iterate on state space and action
  // space, since they are discrete sets. To do so, we can tweak S_ and
  // A_ to provide them with iteration capabilities.

  // For actions, the conversion is obvious, since in C++, bool is
  // mapped naturally on to {0, 1}.
  using A = rl2::enumerable::count<A_,  2>; // A::base_type is A_, i.e. bool.

  // For characters, we need to map {'A', 'B', 'C', ... 'J'} onto {0, 1,
  // ... 9}. This needs a shift (since A is the char value 65). To
  // implement the conversion, we have to define a index convertion
  // static functor.
  struct S_index_convertor {
    static S_          to  (std::size_t index) {return static_cast<S_>(index + 65);         }
    static std::size_t from(S_          value) {return static_cast<std::size_t>(value) - 65;}
  };
  using S = rl2::enumerable::count<S_, 10, S_index_convertor>; // S::base_type is S_, i.e. char.

  // We can also enumerate the Cartesian product of 2 enumerable sets.
  using SA = rl2::enumerable::pair<S, A>;


  template <typename RANDOM_GENERATOR>
  auto build_mdp(RANDOM_GENERATOR& gen, double correct_answer_probability, bool show_reward_table=false) {
  
    // Let us define the reward table for the game. We can have arrays
    // since the state space size S::size is known at compiling time.
    std::array<double, S::size> rewards = {0, 50, 100, 200, 400, 600, 1000, 1500, 3000, 5000};

    // Let us display the reward table. The value of 'it' can be used
    // for accessing the elements of a tabular storing related to S
    // (here the rewards).
    if(show_reward_table) {
      std::cout  << "Rewards : " << std::endl;
      for(auto it = S::begin; it != S::end; ++it)
	std::cout << "  for state " << *it << " : " << rewards[it] << std::endl; 
      std::cout << std::endl;
    }

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


}
