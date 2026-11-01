#include <iostream>
#include <iomanip>
#include <cmath>

#include <rllib2.hpp>

// Digitizing consists in discretizing an interval. Let us see the differnet ways to do this.

constexpr double min_bound {0};
constexpr double max_bound {80};
constexpr std::size_t nb_bins {10};

template<bool AT_MIN, bool AT_MAX>
void basic_print() {
  std::cout << std::boolalpha << AT_MIN << ' ' << AT_MAX << std::endl;
  for(double i = min_bound; i <= max_bound; ++i) {
    auto idx = rl2::enumerable::utils::digitize::to_index<AT_MIN, AT_MAX>(i,   min_bound, max_bound, nb_bins);   // we convert a value into an index.
    auto val = rl2::enumerable::utils::digitize::to_value<AT_MIN, AT_MAX>(idx, min_bound, max_bound, nb_bins); // we get a value from an index;
    char letter {'o'};
    if(std::fabs(val-i) > .5) {
      if(idx % 2 == 0) letter = '=';
      else             letter = '-';
    }
    std::cout << letter;
  }
  std::cout << std::endl << std::endl;
}

int main(int argc, char* argv[]) {

  // Let us display the discretizing
  basic_print<false, false>();
  basic_print<false, true >();
  basic_print<true,  true >();
  basic_print<true,  false>();

  return 0;
}
