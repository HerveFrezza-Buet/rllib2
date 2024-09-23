/*

Copyright 2024 Herve FREZZA-BUET, Alain DUTECH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#pragma once

#include <Eigen/Dense>

namespace rl2 {
  namespace eigen {
    namespace nuplet {
    
      template<std::size_t DIM>
      struct from : public Eigen::Vector<double, DIM> {
	constexpr static std::size_t dim = DIM;
      };
    
    }
  }
}
