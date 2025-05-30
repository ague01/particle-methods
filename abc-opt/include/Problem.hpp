#pragma once

#include "TypeTraits.hpp"

using namespace type_traits;

/**
 * @brief Class that represents a problem to be optimized
 * 
 * @tparam dim indicates the dimension of the space in which the problem is defined
 * 
 * @param fitness_function_ the function to be optimized
 * @param lower_bounds_ the lower bounds of the search space
 * @param upper_bounds_ the upper bounds of the search space
 * @param equality_constraints_ the equality constraints of the problem
 * @param inequality_constraints_ the inequality constraints of the problem
 */
template <size_t dim>
class Problem
{
private:
	RealFunction<dim> fitness_function_;
	RealVector<dim> lower_bounds_;
	RealVector<dim> upper_bounds_;
	std::vector<RealFunction<dim>> equality_constraints_;
	std::vector<RealFunction<dim>> inequality_constraints_;

public:
	/**
	 * @brief Construct a new Problem object
	 */
	Problem(const RealFunction<dim> &fitness_function, const RealVector<dim> &lower_bounds, const RealVector<dim> &upper_bounds)
		: fitness_function_(fitness_function),
		  lower_bounds_(lower_bounds),
		  upper_bounds_(upper_bounds),
		  equality_constraints_(){};                                                 

	const RealFunction<dim> &get_fitness_function() const { return fitness_function_; }
	const RealVector<dim> &get_lower_bounds() const { return lower_bounds_; }
	const RealVector<dim> &get_upper_bounds() const { return upper_bounds_; }
	double get_lower_bound(size_t i) const { return lower_bounds_[i]; }
	double get_upper_bound(size_t i) const { return upper_bounds_[i]; }
	const std::vector<RealFunction<dim>> &get_equality_constraints() const { return equality_constraints_; }
	const std::vector<RealFunction<dim>> &get_inequality_constraints() const { return inequality_constraints_; }

	/**
	 * @brief Add an equality constraint to the problem
	 * 
	 * @param constraint the equality constraint to be added
	 */
	void add_equality_constraint(const RealFunction<dim> &constraint) { equality_constraints_.push_back(constraint); }

	/**
	 * @brief Add an inequality constraint to the problem
	 * 
	 * @param constraint the inequality constraint to be added
	 */
	void add_inequality_constraint(const RealFunction<dim> &constraint) { inequality_constraints_.push_back(constraint); }

	/**
	 * @brief Check if the problem has constraints
	 * 
	 * @return true if the problem has constraints, false otherwise
	 */
	bool has_constraints() const { return !equality_constraints_.empty() || !inequality_constraints_.empty(); }
};