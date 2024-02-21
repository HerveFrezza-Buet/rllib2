#pragma once
#include <array>

#include <rllib2Concepts.hpp>
namespace rl2 {
  namespace nuplet {

    template<concepts::nuplet A, concepts::nuplet B>
    requires (A::dim == B::dim)
    double dot_product(const A& as, const B& bs) {
      double res = 0;
      auto a_ptr = as.begin();
      for(auto b : bs) res += *(a_ptr++) * b;
      return res;
    }
    
    /**
     * @short This wraps a range type so that it provides its size at compilation time.
     * See specializations. Nuplets are used to check dimensions at compiling.
     */

    template<std::ranges::input_range T, std::size_t DIM>
    struct from_range : public T {
      template <typename... Args> from_range(Args&&... args) : T(std::forward<Args>(args)...) {}
      constexpr static std::size_t dim = DIM;
    };
    
    template<typename T, std::size_t DIM>
    struct from : public std::array<T, DIM> {
      constexpr static std::size_t dim = DIM;
    };
    

    /**
     * @short This builds a nuplet instance from a range.
     */
    template<std::size_t DIM, typename T>
    auto make_from_range(T&& source) {return from_range<T, DIM>(std::forward<T>(source));}
    
    /**
     * @short This builds a nuplet instance from an iterator and a size.
     */
    template<std::size_t DIM, std::random_access_iterator ITERATOR>
    auto make_from_iterator(ITERATOR begin) {return make_from_range<DIM>(std::ranges::subrange(begin, begin + DIM));}
  }
}
