#pragma once

#include "discrete-rocket-problem.hpp"

// This configures the rocket discretization.

using types = rocket::enumerable::types<
  101, -200., 200., // nb errors, min, max
  051,  -50.,  50., // nb speeds, min, max
  15.,   .5>;       // up thrust, simulation step duration (dt).

inline gdyn::problem::rocket::parameters make_params() {return {};}
