#pragma once

namespace rl2 {
  namespace nuplet {
    /**
     * @short This wraps a range type so that it provides its size at compilation time.
     */
    template<typename RANGE, std::size_t DIM>
    struct from : public RANGE {
      template <typename... Args> from(Args&&... args) : RANGE(std::forward<Args>(args)...) {}
      constexpr static std::size_t dim = DIM;
    };

    template<std::size_t DIM, typename RANGE>
    auto from_range(RANGE&& source) {return from<RANGE, DIM>(std::forward<RANGE>(source));}
    
    template<std::size_t DIM, std::random_access_iterator ITERATOR>
    auto from_iterator(ITERATOR begin) {return from_range<DIM>(std::ranges::subrange(begin, begin + DIM));}
  }
}
