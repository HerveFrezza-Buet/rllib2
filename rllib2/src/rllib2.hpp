#pragma once

/**
 * @example weakest-link-problem.hpp
 * @example example-001-000-understanding-enumerables.cpp
 * @example example-001-001-getting-started.cpp
 * @example example-001-002-functionals.cpp
 * @example example-001-003-sarsa-qlearning.cpp
 * @example example-001-004-enumerable-systems.cpp
 * @example example-002-001-cartpole-discrete.cpp
 * @example example-003-001-linear-features.cpp
 * @example example-003-002-linear-q.cpp
 */

#include <rllib2Checkings.hpp>
#include <rllib2Critic.hpp>
#include <rllib2Enumerable.hpp>
#include <rllib2Features.hpp>
#include <rllib2Functional.hpp>
#include <rllib2Iterators.hpp>
#include <rllib2MDP.hpp>
#include <rllib2Nuplet.hpp>
#include <rllib2Ranges.hpp>
#include <rllib2Concepts.hpp>
#include <rllib2Transition.hpp>

// Problem definitions
#include <rllib2-defs-mountain-car.hpp>
#include <rllib2-defs-cartpole.hpp>


/*

  TODO : a virer

  Tout ce qui est discret est dans le namespace enumerable
  Tout ce qui est discret seulement en action est dans le namespace enumarable::action (ex discrete_a).

  le concept pour les ensembles finis est concepts::enumerable::finite, mais le template pour les faire, c'est un enumrable::set (on pourrait mettre enumerable::finite_set pas comme tous les sets enumerables qu'on offre sont finis... je n'ai pas trouvé ça utile).

  tabular n'est plus un namespace, mais le nom d'une classe. En effet, les fonctions à arguments énumérables suivent le concept enumerable::two_args_function ou enumerable::action::two_args_function, on n'a pas eu besoin de concept general enumerable::function de fonction qui prendraient un arg qui soit enumérable. Du coup, nous proposons des outils pour construire des fonctions à arguments énumérable codées de façon tabulaires. Mais toute fonction à argument énumerable n'est pas tabulaire. Donc tabular, c'est le nom de l'outil qui fabrique des fonctions enumérables tabulaires, ce n'est ni un concept ni un namespace.

  
 */
