#include <iostream>
#include <iomanip>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read this file.
#include "weakest-link-problem.hpp"

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.
  return 0;
}
