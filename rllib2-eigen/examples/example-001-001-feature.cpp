#include <iostream>
#include <Eigen/Dense>
#include <rllib2-eigen.hpp>

int main(int argc, char* argv[]) {

  {
    auto phi = rl2::eigen::feature::polynomial<8> {};
    Eigen::Vector<double, 9> res = phi(2);
    std::cout << res << std::endl;
  }
  
  return 0;
}
