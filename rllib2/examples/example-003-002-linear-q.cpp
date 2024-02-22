#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <sstream>

#include <rllib2.hpp>

// This shows how linear Q-function are implemented with rllib2.

constexpr unsigned int nb_gauss_x = 3;
constexpr unsigned int nb_gauss_v = 5;
constexpr unsigned int nb_rbfs    = nb_gauss_x * nb_gauss_v;

using S         = rl2::nuplet::from<double,  2>;                                                // s = (x, dx/dt) in Position x Speed
using A         = rl2::enumerable::set<int,  3>;                                                // a in {0, 1, 2}
using rbf       = rl2::functional::gaussian<S>;                                                 // This is our RBF functions type.
using S_feature = rl2::features::rbfs<nb_rbfs, rbf>;                                            // This is our feature type for S.
using params    = rl2::nuplet::from<double, rl2::linear::discrete_a::q_dim_v<S, A, S_feature>>; // This is for theta.
using Q         = rl2::linear::discrete_a::q<params, S, A, S_feature>;                          // functions such as Q(s,a) = thetaT.[0...phi(s)...0]


// Let us display the q-values
void print(std::ostream& os,
	   Q q,               // Yes, I copy q here, it is only a pair of shared pointers.
	   double xmin, double xmax, std::size_t nb_x,
	   double vmin, double vmax, std::size_t nb_v) {
  std::size_t slot = 4;
  std::string sep = " | ";
  std::size_t a_stride = nb_x * (slot) + (nb_x - 1) * sep.size();
  
  std::ostringstream bar;
  for(auto a_it = A::begin(); a_it != A::end(); ++a_it)
    bar << "-+-" << std::string(a_stride, '-');
  bar << "-+-" << std::endl;
  
  os << bar.str();
  for(auto a_it = A::begin(); a_it != A::end(); ++a_it) {
    std::ostringstream a_title;
    a_title << "a = " << static_cast<A::base_type>(a_it);
    os << sep << std::setw(a_stride) << a_title.str();
  }
  os << sep << std::endl;
  os << bar.str();
  for(std::size_t vid = 0; vid < nb_v; ++vid) {
    auto v = rl2::enumerable::utils::digitize::to_value(vid, vmin, vmax, nb_v);
    for(auto a_it = A::begin(); a_it != A::end(); ++a_it) {
      for(std::size_t xid = 0; xid < nb_x; ++xid) {
	auto x = rl2::enumerable::utils::digitize::to_value(xid, xmin, xmax, nb_x);
	os << sep << std::setw(slot) << (unsigned int)(100 * q({x, v}, *a_it) + .5); // Nota: q({x, v}, *a_it)
      }
    }
    os << sep << std::endl;
  }
  os << bar.str();
  
}

int main(int argc, char* argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  
  auto [xmin, xmax, sigma_x] = std::make_tuple(-1. , 10., 3.0);
  auto [vmin, vmax, sigma_v] = std::make_tuple( -.5,  3.,  .2);

  // We set phi with empty radial bases.
  S_feature phi {std::make_shared<S_feature::rbfs_type>()};

  // Let us build the bases...

  // This is the common parameters for the 2D Gaussians.
  auto gammas_ptr = std::make_shared<S>(rl2::functional::gaussian_gammas_of_sigmas<S>({sigma_x, sigma_v}));

  // Let us set the Gaussians
  auto out_it = phi.rbfs->begin();
  for(std::size_t vid = 0; vid < nb_gauss_v; ++vid) {
    auto mu_v = rl2::enumerable::utils::digitize::to_value(vid, vmin, vmax, nb_gauss_v);
    for(std::size_t xid = 0; xid < nb_gauss_x; ++xid) {
      auto mu_x = rl2::enumerable::utils::digitize::to_value(xid, xmin, xmax, nb_gauss_x);
      *(out_it++) = rbf {{mu_x, mu_v}, gammas_ptr}; // We set each gaussian rbf
    }
  }
  
  // ... the bases used by phi are all set up now, sharing gammas_ptr.

  // Now, let is build up a Q function from two shared pointers (one for phi, one for theta).
  
  Q q {std::make_shared<S_feature>(std::move(phi)), std::make_shared<params>()};
  
  // q.s_feature is the features used for s. The phi object has been moved there, so it is invalid now.
  // q.params is the parameter set.
  // both q.s_feature and q.params are shared pointers, you can refer top them safely.

  // Let us initialize q with random parameters.
  rl2::nuplet::random_init(gen, *(q.params), 0, 1);
  print(std::cout, q,
	xmin, xmax, 5,
	vmin, vmax, 5);
  std::cout << std::endl;

  // Let us consider a specific state
  S s {1., 1.}; // (x, v).
  A a {1};

  std::cout << "q(s, a) = " << q(s, a) << std::endl
	    << std::endl; 

  auto q_s = q(s); // Q is partially callable.
  for(auto a_it = A::begin(); a_it != A::end(); ++a_it)
    std::cout << "q_s(" << static_cast<A::base_type>(a_it) << ") = " << q_s(*a_it) << std::endl;
  std::cout << std::endl;
  
  auto greedy_on_q = rl2::discrete::greedy_ify(q); // or rl2::discrete::argmax_ify(q)


  return 0;
}
