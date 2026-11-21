#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <random>
#include <algorithm>

#include <gdyn.hpp>
#include <rllib2.hpp>


// This examples introduces the use of tabular functions, typically
// for V and Q, when the state space and the action space are both
// enumerable.

// Read the type definitions (S, A and SA) in this file.
#include "weakest-link-problem.hpp"


int main(int argc, char* argv[]) {

  std::cout << std::boolalpha;

  {
    // First, let us define a tabular function. It takes values in an
    // enumerable state and produces a result taken from a table.

    std::array<double, weakest_link::S::size()> values_table;
    std::fill(values_table.begin(), values_table.end(), 0.);
    
    auto V = rl2::enumerable::make_tabular<weakest_link::S>(values_table.begin());
    const auto& constV {V}; // The same as V, but read-only.

    // We can use V(s) to read/write the table.
    weakest_link::S sA {'A'};
    weakest_link::S sB {'B'};
    weakest_link::S sJ {'J'};
    weakest_link::S sD {'D'};
    
    rl2::enumerable::index sDidx {sD};
    
    V(sA)    = 1.;
    V(sB)    = 2.;
    V(sJ)    = V(sA) + V(sB);
    V(sDidx) = 9.;
    for(auto it = weakest_link::S::begin(); it != weakest_link::S::end(); ++it)
      std::cout << "V(" << *it << ") = " << V(it) << std::endl;

    std::cout << std::endl
	      << "constV(" << static_cast<weakest_link::S::base_type>(sB) << ") = " << constV(sB) << std::endl;
    // constV(sB) = 10;  // This does not compile.
  }
  
  std::cout << std::endl
	    << "---------------------" << std::endl
	    << std::endl;
  
  {
    std::array<double, weakest_link::SA::size()> values_table;
    std::fill(values_table.begin(), values_table.end(), 0.);

    auto Q = rl2::enumerable::make_two_args_tabular<weakest_link::S, weakest_link::A>(values_table.begin());
    const auto& constQ {Q}; // The same as V, but read-only.

    weakest_link::SA sAT {std::make_pair('A', true)};
    weakest_link::SA sAF {std::make_pair('A', false)};
    weakest_link::SA sBT {std::make_pair('B', true)};
    
    weakest_link::S sD {'D'};
    weakest_link::A aT {true};
    
    rl2::enumerable::index sBtidx {sBT};

    Q(sAF) = 1.;
    Q(sAT) = 2.;
    Q(sBtidx) = 3.;
    Q(sD, aT) = 4.;
    Q('F', true) = 5.;

    // Let us apply Q partially.
    
    // auto QG {Q('G')}; // this fails, since 'G' is a char, and thus
    //                      a number, casted into 71, interpreted as an index (like it was
    //                      for Q(sBtidx)).

    weakest_link::S sG {'G'};
    auto QG {Q(sG)}; // This is a partial application.
    QG(false) = 6.;
    QG(aT) = 7.;

    for(auto it = weakest_link::SA::begin(); it != weakest_link::SA::end(); ++it) {
      auto [s, a] = *it;
      std::cout << "Q(" << s << ", " << std::setw(5) << std::left << a << ") = " << Q(it) << std::endl;
    }

    // Now let us use a const Q.
    
    auto [s, a] = static_cast<weakest_link::SA::base_type>(sBT);
    std::cout << std::endl
	      << "constQ(" <<  s << ", " << std::setw(5) << std::left << a << ") = " << constQ(sBT) << std::endl;
    // constQ(sBT) = 10.; // This does not compile.

    auto constQG {constQ(sG)};
    std::cout << std::endl
	      << "constQG(true ) = " << constQG(true ) << std::endl
	      << "constQG(false) = " << constQG(false) << std::endl;
    // constQG(true) = 10; // This does not compile.
  }

  

  
  
  return 0;
}
