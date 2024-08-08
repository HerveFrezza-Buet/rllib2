#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <ranges>
#include <string>

namespace utils {
namespace trace {
template <char SEP = '\t'> struct csv {
  std::ostream &os;
  csv(std::ostream &os) : os(os){};

  template <std::ranges::range LINE> void operator+=(const LINE &line) {
    std::copy(line.begin(), line.end(),
              std::ostream_iterator<std::ranges::range_value_t<LINE>>(
                  os, std::string(1, SEP).c_str() ));
    os << std::endl;
  }
};
} // namespace trace
} // namespace utils
