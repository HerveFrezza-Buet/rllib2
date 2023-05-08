#pragma once

#include <rllib2Specs.hpp>

namespace rl2 {

  /**
   * This is an orbit SARSA "segment".
   */
  template<typename S, typename A>
  struct sarsa {
    S s;
    A a;
    double r;
    S ss;
    std::optional<A> aa;

    sarsa()                       = default;
    sarsa(const sarsa&)            = default;
    sarsa(sarsa&&)                 = default;
    sarsa& operator=(const sarsa&) = default;
    sarsa& operator=(sarsa&&)      = default;
    
    sarsa(const S& s, const A& a, double r, const S& ss, const std::optional<A> aa)
      : s(s), a(a), r(r), ss(ss), aa(aa) {}
    
    sarsa(const S& s, const A& a, double r, const S& ss)
      : sarsa(s, a, r, ss, std::nullopt) {}
    
    sarsa(const S& s, const A& a, double r, const S& ss, const A& aa)
      : sarsa(s, a, r, ss, aa) {}


    bool is_terminal() const {return !(aa.has_value());}
    
    template<specs::mdp_orbit_point ORBIT_POINT>
    void operator+=(const ORBIT_POINT& next) {
      s  = ss;
      a  = *aa;
      r  = next.observation.second;
      ss = next.observation.first;
      aa = next.command;
    }
  };


  /**
   * This builds a sarsa.
   *
   * @param current An orbit point, that must not be a terminal one.
   * @param next    An orbit point.
   */
  template<specs::mdp_orbit_point ORBIT_POINT>
  auto make_sarsa(const ORBIT_POINT& current, const ORBIT_POINT& next) {
    return sarsa<typename ORBIT_POINT::observation_type::first_type, typename ORBIT_POINT::command_type>(current.observation.first, *current.command, next.observation.second, next.observation.first, next.command);
  }

}
