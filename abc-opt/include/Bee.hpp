#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <random>

#include "Problem.hpp"
#include "TypeTraits.hpp"

using namespace type_traits;

/**
 * @brief Class that represents a single Bee in the colony
 *
 * @param fi_value_ the value of the fitness function in the global best position
 * @param fitness_probability_ proportional to the fitness value of the Bee, used in updating the position
 * @param failure_counter_ the number of iterations in which the Bee has not improved
 * @param constraint_violation_ the contraint violation of the Bee's position
 * @param fitness_value_ the fitness value of the Bee
 * @param index_in_colony_ the index of the Bee in the colony used when updating the position
 *
 * @tparam dim the dimension of the space in which the function is defined
 */
template <size_t dim> class Bee
{
private:
  std::shared_ptr<Problem<dim>> problem_;
  std::shared_ptr<std::mt19937> random_generator_;
  RealVector<dim> position_;
  double cost_value_;
  double fitness_probability_;
  int failure_counter_;
  double constraint_violation_;
  double fitness_value_;
  size_t index_in_colony_;

public:
  /**
   * @brief Construct a new Bee object
   *
   * @param problem shared pointer to the problem to be optimized
   */
  Bee (const std::shared_ptr<Problem<dim>> &problem, const std::shared_ptr<std::mt19937> &random_generator,
       size_t index_in_colony)
      : problem_ (problem), random_generator_ (random_generator), index_in_colony_ (index_in_colony) {};

  Bee () = default;

  /**
   * @brief Initialize the Bee parameters
   */
  void initialize ();

  /**
   * @brief Update the Bee position, using the Employed Bee strategy
   *
   * @param MR modification rate
   * @param colony const reference to colony, needed to read data from other Bees in order to update your
   * position.
   */
  void update_position (const double MR, const std::vector<Bee<dim>> &colony);

  /**
   * @brief Compute the probability of each position to be chosen by employed bees
   *
   * @param total_fitness_value
   * @param total_constraint_violation
   */
  void compute_probability (const double total_fitness_value, const double total_constraint_violation,
                            const double tol = 1e-6);

  /**
   * @brief Print the Bee parameters and actual state
   *
   * @param out the output stream where to write the data
   */
  void print (std::ostream &out = std::cout) const;

  /**
   * @brief Get the fitness value for the acutal position occupied by the Bee.
   *
   * @return double the fitness value of the best position
   */
  double
  get_value () const
  {
    return cost_value_;
  }

  /**
   * @brief Get the current position of the Bee
   *
   * @return const RealVector& the position vector of the Particle
   */
  const RealVector<dim> &
  get_position () const
  {
    return position_;
  }

  /**
   * @brief Get the failure counter of the Bee
   *
   * @return int the failure counter
   */
  int
  get_failure_counter () const
  {
    return failure_counter_;
  }

  /**
   * @brief Get the current total constraint violation of the particle
   *
   * @return double total constraint violation
   */
  double
  get_constraint_violation () const
  {
    return constraint_violation_;
  }

  /**
   * @brief Get the fitness probability of the Bee
   *
   * @return double the fitness probability
   */
  double
  get_probability () const
  {
    return fitness_probability_;
  }

  /**
   * @brief compute the fitness value, useful for the fitness probability computation used by the onlookers
   *
   * @return double consisting in the fitness value
   */
  double compute_fitness_value ();

  /**
   * @brief Utility to get the best position between the given two, following the feasibility-based rule
   *
   * @param value the fitness value on the position you are comparing with the one of this Bee
   * @param viol the constraint violation on the position you are comparing with the one of this Bee
   * @param tol the tolerance to be used in the comparison
   * @return true if this Bee is better than the other one
   * @return false otherwise
   */
  bool feasibility_rule (const double value, const double viol, const double tol = 1e-6) const;

private:
  /**
   * @brief Utility to update the total constraint violaton at the current position
   *
   * @return double indicating the amount of constraint violation
   */
  double compute_constraint_violation (const RealVector<dim> &position) const;
};

#include "Bee.cpp"
