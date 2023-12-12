#include <iostream>
#include <Eigen/Dense>
#include <rllib2-eigen.hpp>

int main(int argc, char* argv[]) {

  {
    auto phi = rl2::eigen::feature::polynomial<8> {};
    Eigen::Vector<double, 9> res = phi(2);
    std::cout << "phi(2): " << std::endl << res << std::endl;

    Eigen::Vector<double, 9> theta {0, 0, 0, 1, 0, 0, 0, 0, 0};
    auto f = rl2::eigen::linear::make<double, 9>(phi, theta);
    auto g = rl2::eigen::linear::make<double, 9>(phi, std::cref(theta));

    
    std::cout << std::endl
	      << f(2) << ", " << f(3) << ", " << g(2) << ", " << g(3) << std::endl;

    theta[0] = 1;

    
    std::cout << std::endl
	      << f(2) << ", " << f(3) << ", " << g(2) << ", " << g(3) << std::endl;
  }
  
  return 0;
}
