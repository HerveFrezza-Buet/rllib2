#include <iostream>
#include <Eigen/Dense>
#include <rllib2-eigen.hpp>
#include <string>

#include <fstream>

// This illustrates ythe use of linear functions, i.e. f(x) = theta^T . phi(x)

using Point2D = Eigen::Vector<double, 2>;

int main(int argc, char* argv[]) {

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    // Let us use a degree 8 polynomial approximation.
    auto phi = rl2::eigen::feature::polynomial<8> {};
    Eigen::Vector<double, 9> res = phi(2); // (2^0, 2^1, 2^2, ... 2^8) 
    std::cout << "phi(2): " << std::endl << res << std::endl;

    // From the previous feature function definition, we can build
    // a linear approximator.
    auto f = rl2::eigen::function::make_linear(phi);
    f.theta = {0, 0, 0, 1, 0, 0, 0, 0, 0}; // we set the parameter.

    std::cout << std::endl
	      << "f(2) = " << f(2) << ", " << "f(3) = " << f(3) << std::endl;
    
    f.theta = {0, 0, 1, 1, 0, 0, 0, 0, 0}; // we modify the parameter.
    
    std::cout << "f(2) = " << f(2) << ", " << "f(3) = " << f(3) << std::endl;

    // NB: size_of(phi) = 0, so make_linear(phi) makes no copy. This
    // will be different with gaussian features.
  }

  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    Point2D x {0, 0};
    
    auto normal = rl2::eigen::function::gaussian(x, 1.0);
    // There is type inference here, the Gaussian works for 2D points.

    std::cout << normal({0., 0.}) << ", " << normal({0.1, 0.1}) << std::endl;
  }

  
  {
    std::cout << std::endl << std::string(10, '-') << std::endl << std::endl;

    // Here, 'double' makes us work with 1D Gaussians.
    rl2::eigen::feature::gaussian_rbf<double, 3> phi {{1.0, .5},  // mean1, sigma1
						      {2.0, .5},  // mean2, sigma2
						      {3.0, .5}}; // mean3, sigma3
    // This is not checked at compiling time...
    if(phi.rbfs.size() != 3)
      std::cout << "ERROR: provide 3 RBFs please..." << std::endl;

    std::cout << std::endl;
    int i = 1;
    for(auto& rbf : phi.rbfs) {
      std::cout << "radial basis #" << i++ << ": rbf(2.0) = " << rbf(2.0) << std::endl;
    }

    std::cout << std::endl
	      << "phi(2.0)" << std::endl
	      << "--------" << std::endl << phi(2.0) << std::endl
	      << std::endl
	      << "phi(2.05)" << std::endl
	      << "---------" << std::endl << phi(2.05) << std::endl;
    
    // We will not use phi anymore, so we can move it for building the linear function.
    auto f = rl2::eigen::function::make_linear(std::move(phi)); 
    f.theta = {0, 1, 1, 1}; // No dimension check is done here...
    
    std::cout << std::endl
	      << "f(2) = " << f(2) << ", " << "f(3) = " << f(3) << std::endl;
    

    // The feature phi is not valid anymore, since we have moved (and
    // not copied) its content into f. Nevertheless, we can retrieve it.
    std::cout << std::endl
	      << "f.phi(2.0)" << std::endl
	      << "----------" << std::endl << f.phi(2.0) << std::endl
	      << std::endl
	      << "f.phi(2.05)" << std::endl
	      << "-----------" << std::endl << f.phi(2.05) << std::endl;

  }
  
  {
    Point2D omega1 {-.5, -.2};
    Point2D omega2 {  0,   1};
    
    // Let us do the same with 2D Gaussians.
    rl2::eigen::feature::gaussian_rbf<Point2D, 2> phi {{omega1, .2},  // mean1, sigma1
						       {omega2, .5}}; // mean2, sigma2
    auto f = rl2::eigen::function::make_linear(std::move(phi)); 
    f.theta = {0, 1, .5};

    // Let us write a gnuplot file.
    std::ofstream gnuplot {"demo.plot"};
    gnuplot << "set hidden3d back" << std::endl
	    << "set xyplane at 0" << std::endl
	    << "splot '-' with lines" << std::endl;
    
    Point2D M;
    for(M[1] = -2; M[1] <= 2; M[1] += .075, gnuplot << std::endl)
      for(M[0] = -2; M[0] <= 2; M[0] += .075)
	gnuplot << M[1] << ' ' << M[0] << ' ' << f(M) << std::endl;

    std::cout << std::endl << std::endl << std::endl
	      << "Run:" << std::endl 
	      << "  gnuplot -p demo.plot" << std::endl
	      << std::endl;
  }


  
  return 0;
}
