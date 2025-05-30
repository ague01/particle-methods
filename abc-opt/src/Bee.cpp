#pragma once

#include "Bee.hpp"

using namespace type_traits;

template <size_t dim>
void
Bee<dim>::initialize ()
{
  // Draw uniformly a value in the range [lower_bound_i, upper_bound_i] for each dimension i
  for (size_t i = 0; i < dim; ++i)
    {
      std::uniform_real_distribution<double> distr (this->problem_->get_lower_bound (i),
                                                    this->problem_->get_upper_bound (i));
      // Initialize the position and velocity vectors
      this->position_[i] = distr (*this->random_generator_);
    }
  // Initialize the failure counter
  failure_counter_ = 0;
  // Initialize the value
  cost_value_ = this->problem_->get_fitness_function () (this->position_);
  // Initialize the constraint violation
  constraint_violation_ = compute_constraint_violation (this->position_);
}

template <size_t dim>
void
Bee<dim>::update_position (const double MR, const std::vector<Bee<dim>> &colony_)
{
  int change_occurred = 0;
  // index reffering to a chosen random neighbour different from the actual bee
  size_t neighbour_index = -1;
  std::uniform_real_distribution<double> distr (0, 1.0);
  std::uniform_real_distribution<double> distrPosUpdate (-1.0, 1.0);
  std::uniform_int_distribution<int> distrNeighbour (0, colony_.size () - 1);
  // new position of the bee after the update
  RealVector<dim> new_position;

  // Update the position of the bee one dimension at a time
  for (int i = 0; i < dim; ++i)
    {
      // Generate a random number in the range [0, 1]
      double r = distr (*this->random_generator_);
      // Update the i-th dimension position's value with probability MR
      if (r < MR)
        {
          change_occurred = 1;
          // Generate a random number in the range [-1, 1], used as a factor to the position update
          double phi = distrPosUpdate (*this->random_generator_);

          // Choose a random neighbour different from the actual bee
          do
            {
              neighbour_index = distrNeighbour (*this->random_generator_);
            }
          while (neighbour_index == index_in_colony_);

          // new position value of the i-th dimension
          double new_parameter
              = this->position_[i] + phi * (this->position_[i] - colony_[neighbour_index].get_position ()[i]);

          // Update the i-th dimension position's value if it is within the bounds, otherwise set it to the
          // bound
          if (new_parameter < this->problem_->get_lower_bound (i))
            new_position[i] = this->problem_->get_lower_bound (i);
          else if (new_parameter > this->problem_->get_upper_bound (i))
            new_position[i] = this->problem_->get_upper_bound (i);
          else
            new_position[i] = new_parameter;
        }
      else
        {
          new_position[i] = this->position_[i];
        }
    }

  if (change_occurred == 0)
    {
      // Choose a random neighbour different from the actual bee
      do
        {
          neighbour_index = distrNeighbour (*this->random_generator_);
        }
      while (neighbour_index == index_in_colony_);

      // Choose a random dimension to be modified
      std::uniform_int_distribution<int> distrRandomDim (0, dim - 1);
      int j = distrRandomDim (*this->random_generator_);

      // Generate a random number in the range [-1, 1], used as a factor to the position update
      double phi = distrPosUpdate (*this->random_generator_);

      // Update position
      double new_parameter
          = this->position_[j] + phi * (this->position_[j] - colony_[neighbour_index].get_position ()[j]);
      if (new_parameter < this->problem_->get_lower_bound (j))
        new_position[j] = this->problem_->get_lower_bound (j);
      else if (new_parameter > this->problem_->get_upper_bound (j))
        new_position[j] = this->problem_->get_upper_bound (j);
      else
        new_position[j] = new_parameter;
    }

  // Compute the new position's behaviour
  double new_position_value_ = this->problem_->get_fitness_function () (new_position);
  double new_position_constraint_violation_ = compute_constraint_violation (new_position);

  // Update the position if the new position is better
  if (!feasibility_rule (new_position_value_, new_position_constraint_violation_))
    {
      this->position_ = new_position;
      this->cost_value_ = new_position_value_;
      this->constraint_violation_ = new_position_constraint_violation_;
      failure_counter_ = 0;
    }
  else
    {
      failure_counter_++;
    }
}

template <size_t dim>
void
Bee<dim>::compute_probability (const double total_fitness_value, const double total_constraint_violation,
                               const double tol)
{
  // If the solution is feasible, the probability is proportional to the fitness value
  if (constraint_violation_ < tol)
    {
      fitness_probability_ = 0.5 + 0.5 * (fitness_value_ / total_fitness_value);
    }
  // If the solution is infeasible, the probability is inversly proportional to the constraint violation
  else
    {
      fitness_probability_ = 0.5 * (1 - (constraint_violation_ / total_constraint_violation));
    }
}

template <size_t dim>
double
Bee<dim>::compute_constraint_violation (const RealVector<dim> &position) const
{
  double violation = 0.0;
  if (this->problem_->has_constraints ())
    {
      for (RealFunction<dim> constraint : this->problem_->get_equality_constraints ())
        violation += std::abs (constraint (position));
      for (RealFunction<dim> constraint : this->problem_->get_inequality_constraints ())
        violation += std::max (0.0, constraint (position));
    }

  return violation;
}

template <size_t dim>
bool
Bee<dim>::feasibility_rule (const double value, const double viol, const double tol) const
{
  // (a) a feasible solution is preferred over an infeasible solution
  if (this->constraint_violation_ < tol && viol > tol)
    return true;
  else if (this->constraint_violation_ > tol && viol < tol)
    return false;
  // (b) among two feasible solutions, the one with better objective function value is preferred
  else if (this->constraint_violation_ < tol && viol < tol)
    return this->cost_value_ < value;
  // (c) among two infeasible solutions, the one with smaller TAV is chosen
  else
    return this->constraint_violation_ < viol;
}

template <std::size_t dim>
void
Bee<dim>::print (std::ostream &out) const
{
  out << "Position: (";
  for (std::size_t i = 0; i < dim; ++i)
    out << this->position_[i] << ", ";
  out << "\b\b)" << std::endl;

  out << "Constraint violation:\t" << constraint_violation_ << std::endl;
}

template <std::size_t dim>
double
Bee<dim>::compute_fitness_value ()
{
  if (cost_value_ >= 0)
    fitness_value_ = 1.0 / (1.0 + cost_value_);
  else
    fitness_value_ = 1.0 + std::abs (cost_value_);
  return fitness_value_;
}