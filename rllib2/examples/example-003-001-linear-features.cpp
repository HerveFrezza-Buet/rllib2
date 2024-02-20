
#include <string>
#include <iostream>
#include <vector>

#include <rllib2.hpp>

// This illustrates the use of linear functions, i.e. f(x) = theta^T . phi(x)


int main(int argc, char* argv[]) {

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    // Let us use a degree 8 polynomial approximation.
    rl2::features::polynomial<8> phi {};

    // ph(x) is a vector. Here, we rely on ranges, so phi(x) is a range containing the vector components.
    std::cout << "phi(2) = [";
    for(double value : phi(2)) std::cout << ' ' << value;
    std::cout << ']' << std::endl;

    // The range_based dot product can thus be done like this (it may
    // not really be invoked direclty when using rllib2).
    std::vector<double> big_theta = {
      1., 2., 3., // These won't be used.
      10000., 0., 0., 1., 0., 0., 0., 0., 0., // We use these 9 ones, i.e. we compute 10000 + x^3
      1., 2., 3., 4., 5.}; // These won't be used.

    // Let us compute a parameter from [big_theta.begin() + 3, big_theta.begin() + 3 + 9]
    auto theta = rl2::nuplet::from_iterator<phi.dim>(big_theta.begin() + 3);
    std::cout << rl2::linear::dot_product(theta, phi(2)) << ' '
	      << rl2::linear::dot_product(theta, phi(3)) << ' '
	      << rl2::linear::dot_product(theta, phi(4)) << ' '
	      << rl2::linear::dot_product(theta, phi(5)) << std::endl;
      
  }

  return 0;
}
  

