/*
  Tentative pour avoir Q(s,a) = (Q(s))(a) sans passer par les pairs.

*/

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "rllib2.hpp"

// State are letters
struct S_index_convertor {
  static char to (std::size_t index) {
    return static_cast<char>(index + 65);
  }
  static std::size_t from(char value) {
    return static_cast<std::size_t>(value) - 65;
  }
};
using S = rl2::enumerable::count<char, 5, S_index_convertor>;

// Actions are Left, Right
struct A_index_convertor {
  static std::string to (std::size_t index) {
    switch(index) {
    case 0: return "Left";
    default: return "Right";
    }
  }
  static std::size_t from(const std::string& value) {
    if (value == "Left") {
      return 0;
    }
    return 1;
  }
};
using A = rl2::enumerable::count<std::string, 2, A_index_convertor>;


using QSVAL_TYPE = std::array<double, A::size>;
using QSFUN_TYPE = rl2::tabular::function<A, QSVAL_TYPE::iterator>;
using QVALDATA_TYPE = std::vector<QSVAL_TYPE>;
//using QVAL_TYPE  = std::array<QSFUN_TYPE, S::size>;
using QVAL_TYPE  = std::vector<QSFUN_TYPE>;

// TODO je pensais que cela passerait par un move
//auto make_qsval(const S& s)
QSVAL_TYPE make_qsval(const S& s)
{
  QSVAL_TYPE qsval;
  qsval.fill( static_cast<std::size_t>(s) );
  return qsval;
}
void print_qsval(const QSVAL_TYPE& q, const S& s) {
  for (auto ita = A::begin; ita != A::end; ++ita ) {
     std::cout << "Q(" << static_cast<S::base_type>(s) << ", ";
     std::cout << static_cast<A::base_type>(*ita) << ")";
     std::cout << " = " << q[static_cast<std::size_t>(ita)] << std::endl;
  }
}
void print_qsfun(const QSFUN_TYPE& q, const S& s) {
  for (auto ita = A::begin; ita != A::end; ++ita ) {
     std::cout << "Q(" << static_cast<S::base_type>(s) << ", ";
     std::cout << static_cast<A::base_type>(*ita) << ")";
     std::cout << " = " << q(*ita) << std::endl;
  }
}
int main(int argc, char *argv[])
{

  // For a given s={B}, lets fill QSVAL and QSFUN
  std::cout << "__qsval('B', .)" << std::endl;
  S s{'B'};
  QSVAL_TYPE qsval;
  qsval.fill( static_cast<std::size_t>(s) );
  print_qsval(qsval, s);

  std::cout << "__qsv2('D', .)" << std::endl;
  auto qsv2 = make_qsval(S{'D'});
  print_qsval(qsv2, S{'D'});

  std::cout << "__(qsfun('D'))(.)" << std::endl;
  QSFUN_TYPE qsfun = rl2::tabular::make_function<A>(qsv2.begin());
  print_qsfun(qsfun, S{'D'});

  // Now, create all functions
  QVAL_TYPE qval;
  QVALDATA_TYPE qval_data;

  for( auto its=S::begin; its != S::end; ++its ) {
    // auto qsv = make_qsval(*its);
    // qval_data.push_back( std::move(qsv) );
    qval_data.emplace_back();
    qval_data.back().fill( static_cast<std::size_t>(its) );
    std::cout << "qsval is created at address " << &(qval_data.back()) << std::endl;
    std::cout << "  pointing at " << &(*(qval_data.back().begin())) << std::endl;
    // auto qsfun = rl2::tabular::make_function<A>(qval_data.back().begin());
    // qval.push_back( std::move(qsfun) );
    qval.emplace_back(qval_data.back().begin());
    std::cout << "qsfun created" << std::endl;
    std::cout << "  pointing at " << *(qval.back().params_it) << "@" << &(*(qval.back().params_it)) << std::endl;
    std::cout << "pushing *****" << std::endl;
    print_qsval(qval_data.back(), *its);
    print_qsfun(qval.back(), *its);
  }
  for( const auto& qs: qval_data) {
    print_qsval(qs, S{'A'});
  }
  std::cout << "First function" << std::endl;
  auto qfun = qval.front();
  std::cout << "  pointing at " << *(qfun.params_it) << "@" << &(*(qfun.params_it)) << std::endl;

  std::cout << "First Values" << std::endl;
  auto qdata = qval_data.front();
  std::cout << "  pointing at " << &(*(qdata.begin())) << std::endl;

  auto qs_first = qfun(A{"Left"});
  std::cout << "  Q(A,Left) = " << qs_first << std::endl;

  std::cout << "__qval" << std::endl;
  for ( auto its=S::begin; its != S::end; ++its) {
    print_qsfun( qval[static_cast<std::size_t>(its)], *its);
  }

  std::cout << "__q" << std::endl;
  auto q = rl2::tabular::make_function<S>(qval.begin());
  std::cout << q(S{'A'})(A{"Left"}) << std::endl;
  std::cout << "  print_qsfun" << std::endl;
  for ( auto its=S::begin; its != S::end; ++its) {
    print_qsfun(q(*its), *its);

    // for (auto ita = A::begin; ita != A::end; ++ita ) {
    //   std::cout << "Q(" << static_cast<S::base_type>(*its) << ", ";
    //   std::cout << static_cast<A::base_type>(*ita) << ")";
    //   std::cout << " = " << (q(*its))(*ita) << std::endl;
    // }
  }

  return 0;
}
