#include <iostream>
#include <iomanip>
#include <array>
#include <random>

#include <gdyn.hpp>
#include <rllib2.hpp>

// Read the type definitions (S, A and SA) in this file.
#include "weakest-link-problem.hpp"

// This is a foreword example, illustrating the use of enumerable
// states. When a set is discrete, its finite number of values can be
// indexed continuously by unsigned integers, starting from 0. The
// index is typically used to set up some tabular function, whose
// results are stored in an array.

// Let us see here how this is managed thanks to enumerable types in
// rllib2.


int main(int argc, char* argv[]) {

  std::cout << std::boolalpha; // Prints booleans as true/false rather than 1/0.

  // Enumerable types have a base type, whose values are associated to
  // an index. Variables of enumarable types can be initialized and
  // affected with a base type value as well as an index value.

  // Enumerables can be constructed from a base_type value or from a
  // size_t index value.
  weakest_link::S s1 {'C'};
  weakest_link::S s2 {std::size_t(5)};
  
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s1) << " has index " << static_cast<std::size_t>(s1) << std::endl;
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;

  // The same stands for affectation.
  s2 = 'H';
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;
  s2 = std::size_t(0);
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s2) << " has index " << static_cast<std::size_t>(s2) << std::endl;

  // Iterators enable to span all the enumerable state values.
  auto it = weakest_link::S::begin(); // 'it' refers to the first value of S.
  ++it;               // 'it' is now the second value of S.
  ++it;               // 'it' is now the third value of S.
  weakest_link::S s3 {it};          // Enumerables can be constructed from an iterator.
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s3) << " has index " << static_cast<std::size_t>(s3) << std::endl;
  std::cout << "State " << *it                           << " has index " << static_cast<std::size_t>(it) << std::endl;
    
  ++it;     // 'it' is now the fourth value of S.
  ++it;     // 'it' is now the fifth value of S.
  s3 = it;  // affectation with iterators is licit.
  std::cout << "State " << static_cast<weakest_link::S::base_type>(s3) << " has index " << static_cast<std::size_t>(s3) << std::endl;


  // If the base_type (char here) can be naturally casted to size_t,
  // you may write some code that compiles fine (no typing error)
  // while it is wrong. This shows the issue.
  char c = 'D';
  std::cout << "The index of " << c << " is "
	    << static_cast<std::size_t>(weakest_link::S(c)) // good conversion to 3
	    << " and not "
	    << static_cast<std::size_t>(c)    // bad (but licit) conversion to 68, be careful !
	    << "! Be careful." << std::endl;
  std::cout << std::endl;

  // enumerable spaces can be iterated. The iterator 'it' can be
  // casted to a std::size_t value, providing us with the index of a
  // particular space value, and *it has the type base_type of the set
  // that has been intrumented to be enumerable.

  std::cout << "State space : " << weakest_link::S::size() << " values" << std::endl;
  for(auto it = weakest_link::S::begin(); it != weakest_link::S::end(); ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;
  
  std::cout << "Action space : " << weakest_link::A::size() << " values" << std::endl;
  for(auto it = weakest_link::A::begin(); it != weakest_link::A::end(); ++it)
    std::cout << "  " << static_cast<std::size_t>(it) << " : " << *it << std::endl;
  std::cout << std::endl;
    
  std::cout << "Cartesian product : " << weakest_link::SA::size() << " values" << std::endl;
  for(auto it = weakest_link::SA::begin(); it != weakest_link::SA::end(); ++it)  {
    auto [s, a] = *it;
    std::cout << "  " << std::setw(2) << static_cast<std::size_t>(it) << " : (" << s << ", " << a << ')' << std::endl;
  }
  std::cout << std::endl;

  auto sa = std::make_pair('G', false); // This is the base_type of SA.
  std::cout << "The pair (" << sa.first << ", " << sa.second << ") has index " << static_cast<std::size_t>(weakest_link::SA(sa)) << '.' << std::endl;

  weakest_link::SA s15 {std::size_t(15)};
  std::cout << "Index 15 : (" << static_cast<weakest_link::SA::base_type>(s15).first << ", " << static_cast<weakest_link::SA::base_type>(s15).second << ')' << std::endl;
  
  return 0;
}
