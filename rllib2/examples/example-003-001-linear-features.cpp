
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

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
    auto theta = rl2::nuplet::from_iterator<phi.dim>(big_theta.begin() + 3);
    std::cout << rl2::linear::dot_product(theta, phi(2)) << ' '
	      << rl2::linear::dot_product(theta, phi(3)) << ' '
	      << rl2::linear::dot_product(theta, phi(4)) << ' '
	      << rl2::linear::dot_product(theta, phi(5)) << std::endl;
      
  }

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    using rbf_feature = rl2::features::rbf<3, rl2::functional::gaussian<double>>;
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

    // Let us check this
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

  return 0;
}
  

