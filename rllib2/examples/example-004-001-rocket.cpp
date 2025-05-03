#include <iostream>

#include <gdyn.hpp>
#include <rllib2.hpp>

#include "discrete-rocket-problem.hpp"

using types = rocket::enumerable::types<
  101, -200., 200., // nb errors, min, max
  051,  -50.,  50., // nb speeds, min, max
  15.,   .5>;       // up thrust, simulation step duration (dt).

int main(int argc, char* argv[]) {

  gdyn::problem::rocket::parameters params;
  auto rocket = gdyn::problem::rocket::system(params);
  auto relative_rocket = gdyn::problem::rocket::relative::system(rocket, [target = params.ceiling_height/2](){return target;});
  auto exposed_rocket  = types::exposed_system(relative_rocket);
  auto discrete_rocket = types::discrete_system(exposed_rocket);


  return 0;
}

  
