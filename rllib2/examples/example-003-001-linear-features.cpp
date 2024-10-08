
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <tuple>
#include <algorithm>

#include <rllib2.hpp>

// This illustrates the use of linear functions, i.e. f(x) = theta^T . phi(x)


// This defines a car. This (and the wrapper below) will be used
// at the end of this example file, skip it for first steps.
    
struct car {
  std::string brand;   
  std::string model;   
  unsigned int  cost;
  unsigned int  miles;
  double motor_power;
};

#define COST_COEF 1e-6
#define MILES_COEF 1e-3
struct car_wrapper {
  // For using RBFs, we need to associate a vector (iterable) to
  // each car. This is what nuplet wrapper is designed for.

  // CONCEPTS: This is required by rl2::concepts::nuplet_wrapper.
  constexpr static std::size_t dim = 4; // This tells how many scalars are used to represent the car.
  
  std::array<double, dim> data;         // This is a buffer for storing them when needed.
      
  // CONCEPTS: This is required by rl2::concepts::nuplet_wrapper.
  car_wrapper(const car& some_car) {

    // Here, we define how a car is represented as 4 scalars, we store
    // the representation of some_car into data.
    
    auto it = data.begin();
    *(it++) = (double)(some_car.cost) * COST_COEF;
    *(it++) = (double)(some_car.miles) * MILES_COEF;
    *(it++) = (double)(some_car.motor_power);

    // Last component is how the brand is perceived by
    // jet-setters, in a scale from 0 (bad) to 100 (excellent). No
    // information is 0... Of course, this is fake here.
    if      (some_car.brand == "Ferrari") *(it++) = 100.;
    else if (some_car.brand == "Bentley") *(it++) =  95.;
    else if (some_car.brand == "Porsche") *(it++) =  90.;
    else if (some_car.brand == "Tesla")   *(it++) =  80.;
    else if (some_car.brand == "Renault") *(it++) =  10.;
    else if (some_car.brand == "Citroen") *(it++) =  10.;
    else if (some_car.brand == "Fiat")    *(it++) =  10.;
    else                                  *(it++) =   0.;

    // We we have 4 scalars for a car.
  }
  
  // CONCEPTS: These are required by
  // rl2::concepts::nuplet_wrapper. They provide the begin-sentinel
  // pair for iterating on the scalars representing our car.
  auto begin() const {return data.begin();}
  auto end()   const {return data.end();}
};


unsigned int uint_1000(double value) {return (unsigned int)(1000 * value + .5);}

// This shows the use of polynomial features.
void test_polynomial() {
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

// This shows the use of rbf features, with 1D gaussians.
void test_1d_gaussian() {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    using rbf_feature = rl2::features::rbfs<3, rl2::functional::gaussian<double>>;
    
    // Let us use a RBF with 3 Gaussians. The input is a scalar.
    rbf_feature phi {};

    // RBFs handle a set of functions. For efficiency reasons, it has
    // to be allocated in the heap externally, as a shared pointer,
    // and then provided to phi.
    phi.rbfs = std::make_shared<rbf_feature::rbfs_type>();
    std::size_t i = 0;

    // we use sigma = 0.1, gamma is 1/(2*sigma^2).
    double gamma = rl2::functional::gaussian_gamma_of_sigma(.1); 
    for(auto& gaussian : *(phi.rbfs)) {
      // We set the means, spanned over [0, 1].
      gaussian.mu = rl2::enumerable::utils::digitize::to_value(i++, 0., 1., rbf_feature::nb_rbfs); 
      gaussian.gamma = gamma;
      std::cout << "rbf " << gaussian << " added." << std::endl;
    }
    std::cout << std::endl;
    
    for(double x = 0; x < 1.1; x += .2) {
      std::cout << "phi(" << std::setw(3) << x << ") = [";
      for(double value : phi(x)) std::cout << ' ' << std::setw(4) << uint_1000(value);
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
      for(double value : phi(x)) std::cout << ' ' << std::setw(4) << uint_1000(value);
      std::cout << ']' << std::endl;
      std::cout << "psi(" << std::setw(3) << x << ") = [";
      for(double value : psi(x)) std::cout << ' ' << std::setw(4) << uint_1000(value);
      std::cout << ']' << std::endl;
      std::cout << "chi(" << std::setw(3) << x << ") = [";
      for(double value : chi(x)) std::cout << ' ' << std::setw(4) << uint_1000(value);
      std::cout << ']' << std::endl;
      std::cout << std::endl;
    }
}

// This is for N-dimentional Gaussians. We need to handle share
// pointers here, in order to keep memory allocation as reduced as
// possible.
void test_nd_gaussian() {
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
    using pos_speed = std::array<double, 2>;                    // This is X x V, an std::array is used (see next for a non-range type).
    using mu_type = rl2::nuplet::from<double, 2>;               // This is the radial centers, it must be a nuplet.
    using rbf = rl2::functional::gaussian<mu_type, pos_speed>;  // This is our RBF functions type. 
    using rbf_feature = rl2::features::rbfs<nb_rbfs, rbf>;      // This is our feature type.
    // Nota: nb_rbfs is used as a template parameter, this is why
    // previous constexpr definitions are mandatory.
    
    mu_type sigmas {sigma_x, sigma_v}; // This is our std_dev in each component.
    // This will be used (and thus shared) by all the rbf functions. We compute this once, here.
    auto gammas_ptr = std::make_shared<mu_type>(rl2::functional::gaussian_gammas_of_sigmas(sigmas));

    // This is as previously
    rbf_feature phi {};
    phi.rbfs = std::make_shared<rbf_feature::rbfs_type>();
    auto out_it = phi.rbfs->begin();
    // We span the X x V space
    for(std::size_t vid = 0; vid < nb_gauss_v; ++vid) {
      auto mu_v = rl2::enumerable::utils::digitize::to_value(vid, vmin, vmax, nb_gauss_v);
      for(std::size_t xid = 0; xid < nb_gauss_x; ++xid) {
	auto mu_x = rl2::enumerable::utils::digitize::to_value(xid, xmin, xmax, nb_gauss_x);
	*(out_it++) = {{mu_x, mu_v}, gammas_ptr}; // We set each gaussian rbf
      }
    }

    std::vector<pos_speed> points {{xmin, vmin}, {0., 0.}, {3., 0.}, {3., 1.}, {3., 2.}, {xmax, vmax}};

    for(const auto& point : points) {
      std::cout << "phi(" << point[0] << ", " << point[1] << ") = [";
      auto values = phi(point);
      auto value_it = values.begin();
      std::cout << ' ' << std::setw(4) << uint_1000(*(value_it++)) << std::endl; // The offset
      for(std::size_t vid = 0; vid < nb_gauss_v; ++vid, std::cout << std::endl)
	for(std::size_t xid = 0; xid < nb_gauss_x; ++xid) 
	  std::cout << ' ' << std::setw(4) << uint_1000(*(value_it++));
      std::cout << ']' << std::endl << std::endl;
    }
}


// This is where the class definitions at the beginning of this file
// are used. We customize the use of Gaussian RBFs to an entry type
// which is not naturally iterable.
void test_custom_gaussian() {
    // Let us illustrate here the need for nuplet adaptor, when the
    // type of the gaussian argument is not naturally iterable.
    // Have a look at car and car_wrapper definitions at the beginning of this file.

    car alices {"Ferrari", "F-40",    2600000,  1000, 478};
    car bobs   {"Porsche", "Carrera",  158000, 50000, 612};
    car mine   {"Renault", "Clio",      13000,     0,  80};

    // Ok, now we can set up RBFs features, as previously 
    using mu_type = rl2::nuplet::from<double, car_wrapper::dim>; 
    using rbf = rl2::functional::gaussian<mu_type, car, car_wrapper>;        
    using rbf_feature = rl2::features::rbfs<3, rbf>;  // This is our feature type, with 3 rbfs.
    
    mu_type sigmas {1., 1., 500., 10.}; // This is our std_dev in each component (cost, miles, power, brand).
    
    // This will be used (and thus shared) by all the rbf functions. We compute this once, here.
    auto gammas_ptr = std::make_shared<mu_type>(rl2::functional::gaussian_gammas_of_sigmas(sigmas));
    
    rbf_feature phi {};
    phi.rbfs = std::make_shared<rbf_feature::rbfs_type>();

    // Let us use pour 3 cars as rbf centers.
    auto out_it = phi.rbfs->begin();
    mu_type center;
    for(const auto& c : {alices, bobs, mine}) {
      car_wrapper wrapper {c};
      std::copy(wrapper.begin(), wrapper.end(), center.begin()); 
      *(out_it++) = {center, gammas_ptr};
    }

    // Let us compute the features for each car
    for(const auto& c : {alices, bobs, mine}) {
      for(auto value : phi(c)) std::cout << value << ' ';
      std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
  test_polynomial();
  test_1d_gaussian();
  test_nd_gaussian();
  test_custom_gaussian();
  return 0;
}
  

