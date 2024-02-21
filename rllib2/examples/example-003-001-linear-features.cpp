
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <tuple>

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
    auto theta = rl2::nuplet::make_from_iterator<phi.dim>(big_theta.begin() + 3);
    std::cout << rl2::nuplet::dot_product(theta, phi(2)) << ' '
	      << rl2::nuplet::dot_product(theta, phi(3)) << ' '
	      << rl2::nuplet::dot_product(theta, phi(4)) << ' '
	      << rl2::nuplet::dot_product(theta, phi(5)) << std::endl;
      
  }

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    using rbf_feature = rl2::features::rbfs<3, rl2::functional::gaussian<double>>;
    
    // Let us use a RBF with 3 Gaussians. The input is a scalar.
    rbf_feature phi {};

    // RBFs handle a set of functions. For efficiency reasons, it has
    // to be allocated in the heap externally, as a shared pointer,
    // and then provided to phi.
    phi.rbfs = std::make_shared<rbf_feature::rbfs_type>();
    std::size_t i = 0;
    double gamma = rl2::functional::gaussian_gamma_of_sigma(.1); // we use sigma = 0.1, gamma is 1/(2*sigma^2).
    for(auto& gaussian : *(phi.rbfs)) {
      gaussian.mu = rl2::enumerable::utils::digitize::to_value(i++, 0., 1., rbf_feature::nb_rbfs); // We set the means spread in [0, 1].
      gaussian.gamma = gamma;
      std::cout << "rbf " << gaussian << " added." << std::endl;
    }
    std::cout << std::endl;
    
    for(double x = 0; x < 1.1; x += .2) {
      std::cout << "phi(" << std::setw(3) << x << ") = [";
      for(double value : phi(x)) std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * value + .5);
      std::cout << ']' << std::endl;
    }
    std::cout << std::endl;

    // We can have another feature by copy, but they will share the rbfs.
    rbf_feature psi {phi};

    // If you want an actual copy of the rbfs, you need to be explicit.
    rbf_feature chi;
    chi.rbfs = std::make_shared<rbf_feature::rbfs_type>(*(phi.rbfs));

    // Let us check this... both phi and psi are identical, and have
    // been modified from the initial phi. The chi function is
    // identical to the initial phi, even after the change.
    auto& rbfs = *(phi.rbfs);
    rbfs[0].mu = .2;
    rbfs[2].mu = .6;

    for(double x = 0; x < 1.1; x += .2) {
      std::cout << "phi(" << std::setw(3) << x << ") = [";
      for(double value : phi(x)) std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * value + .5);
      std::cout << ']' << std::endl;
      std::cout << "psi(" << std::setw(3) << x << ") = [";
      for(double value : psi(x)) std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * value + .5);
      std::cout << ']' << std::endl;
      std::cout << "chi(" << std::setw(3) << x << ") = [";
      for(double value : chi(x)) std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * value + .5);
      std::cout << ']' << std::endl;
      std::cout << std::endl;
    }
  }

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    // Let us do the same as previously, with vectors rather that
    // scalar inputs. For example, let us suppose that we want to deal
    // with a problem where we have to represent the scalar position x
    // and scalar speed v of some 1D stuff (e.g mountain car).
    constexpr unsigned int nb_gauss_x = 3;
    constexpr unsigned int nb_gauss_v = 5;
    constexpr unsigned int nb_rbfs    = nb_gauss_x * nb_gauss_v;
    auto [xmin, xmax, sigma_x] = std::make_tuple(-1. , 10., 3.0);
    auto [vmin, vmax, sigma_v] = std::make_tuple( -.5,  3.,  .2);

    // Let us define our types, based on an std::array behind the scene for storing x and v.
    using pos_speed = rl2::nuplet::from<double, 2>;        // This is X x V, an std::array is used.
    using rbf = rl2::functional::gaussian<pos_speed>;      // This is our RBF functions type.
    using rbf_feature = rl2::features::rbfs<nb_rbfs, rbf>; // This is our feature type.
    // Nota: nb_rbfs is used as a template parameter, this is why
    // previous constexpr definitions are mandatory.
    
    pos_speed sigmas {sigma_x, sigma_v}; // This is our std_dev in each component.
    // This will be used (and thus shared) by all the rbf functions. We compute this once, here.
    auto gammas_ptr = std::make_shared<pos_speed>(rl2::functional::gaussian_gammas_of_sigmas(sigmas));

    // This is as previously
    rbf_feature phi {};
    phi.rbfs = std::make_shared<rbf_feature::rbfs_type>();
    auto out_it = phi.rbfs->begin();
    // We span the X x V space
    for(std::size_t vid = 0; vid < nb_gauss_v; ++vid) {
      auto mu_v = rl2::enumerable::utils::digitize::to_value(vid, vmin, vmax, nb_gauss_v);
      for(std::size_t xid = 0; xid < nb_gauss_x; ++xid) {
	auto mu_x = rl2::enumerable::utils::digitize::to_value(xid, xmin, xmax, nb_gauss_x);
	*(out_it++) = rbf {{mu_x, mu_v}, gammas_ptr}; // We set each gaussian rbf
      }
    }

    std::vector<pos_speed> points {{xmin, vmin}, {0., 0.}, {3., 0.}, {3., 1.}, {3., 2.}, {xmax, vmax}};

    for(const auto& point : points) {
      std::cout << "phi(" << point[0] << ", " << point[1] << ") = [";
      auto values = phi(point);
      auto value_it = values.begin();
      std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * *(value_it++) + .5) << std::endl; // The offset
      for(std::size_t vid = 0; vid < nb_gauss_v; ++vid, std::cout << std::endl)
	for(std::size_t xid = 0; xid < nb_gauss_x; ++xid) 
	  std::cout << ' ' << std::setw(4) << (unsigned int)(1000 * *(value_it++) + .5);
      std::cout << ']' << std::endl << std::endl;
    }
    
  }


  return 0;
}
  

