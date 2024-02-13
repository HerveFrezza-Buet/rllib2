#include <iostream>
#include <Eigen/Dense>
#include <rllib2-eigen.hpp>
#include <string>

#include <fstream>


// The approximate Q evaluation method sometimes use linear
// parametrized Q functions. Usually, the action space remains
// discrete, and the feature only consists of multimplexing (for each
// action) a state feature on the appropriate dimension. This is what
// the discrete-a namespace refers to.

using S = double;

struct A_index_convertor { // ascii('X') = 88
  static char          to(std::size_t index) {return static_cast<char>(index + 88);}        
  static std::size_t from(char        value) {return static_cast<std::size_t>(value) - 88;}
};
using A = rl2::enumerable::set<char, 3, A_index_convertor>; // A = {'X', 'Y', 'Z'}


int main(int argc, char* argv[]) {

  // We use 3 gaussian to "featurize" S.
  rl2::eigen::feature::gaussian_rbf<S, 3> s_phi {{.25, .2},  // mean1, sigma1
						 {.50, .2},  // mean2, sigma2
						 {.75, .2}}; // mean3, sigma3
  
  // Let us build a linear function : S x A -> double
  auto q = rl2::eigen::feature::discrete_a::make_linear<A>(std::move(s_phi)); // s_phi will not be re-usable after that move.
  q.theta <<
    0,   1,   1,   1,  // for a = 0
    0,  10,  10,  10,  // for a = 1
    0, 100, 100, 100;  // for a = 2

  std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

  S s = 0.5;
  std::cout << "### s_phi(" << s << ")" << std::endl
	    << q.s_phi(s) << std::endl
	    << "Thus sum is 1 + " << q.s_phi(s)[1] << " + " << q.s_phi(s)[2] << " + " << q.s_phi(s)[3]
	    << " = 1 + " << q.s_phi(s).sum() - 1.0 << std::endl
	    << std::endl;

  for(auto a_it = A::begin(); a_it != A::end(); ++a_it) 
    std::cout << "Q(" << s << ", " << *a_it << ") = " << q(s, a_it) << std::endl; // a_it is converted into a A.
  std::cout << std::endl
	    << q(s, 'Y') << " = " << q(s, (std::size_t)1) << std::endl
	    << std::endl;

  // We can also use pairs as arguments.
  for(s = 0; s <= 1.01; s += .05)
    std::cout << q({s, 'Z'}) << std::endl;

  return 0;
}
