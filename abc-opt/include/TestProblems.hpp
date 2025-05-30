#pragma once

#define _USE_MATH_DEFINES

#include <functional>
#include <iostream>
#include <array>
#include <math.h>
#include <cmath>

#include "Problem.hpp"

/**
 * @brief Namespace that contains some optimization problem used for the library testing
 *
 */
namespace TestProblems
{
	/**
	 * @brief Problems that can be used for testing purposes.
	 * - TOWNSEND: 2D function with global minimum f(2.0052938, 1.1944509) = -2.0239884
	 * - GOMEZ_LEVY: 2D function with global minimum f(0.089842010, -0.7126564) = -1.031628453
	 * - G10: 8D function with global minimum f(579.3167,1359.943,5110.71,182.0714,295.5985,217.979,286.4162,395.5979) = 7049.248
	 * - G7: 10D function with global minimum f(2.1719, 2.3636, 8.7739, 5.0959, 0.9906, 1.4305, 1.3216, 9.8287, 8.2800, 8.3759) = 24.30620
	 * - GRIEWANK: 30D function with global minimum f(0,...,0) = 1
	 */
	enum ProblemName
	{
		TOWNSEND,
		GOMEZ_LEVY,
		G10,
		G7,
		GRIEWANK
	};

	/**
	 * @brief Create and return the specified testing problem
	 *
	 * @tparam dim the dimension of the space in which the problem is defined
	 * @param p the name of the problem to be returned
	 * @return Problem<double> required problem object
	 */
	template <size_t dim>
	Problem<dim> create_problem(ProblemName p);

	/**
	 * @brief Get the exact value solving the minimization of the given problem
	 *
	 * @tparam dim the dimension of the space in which the problem is defined
	 * @param p the name of the problem for which the solution is required
	 * @return double the exact solution
	 */
	template <size_t dim>
	double get_exact_value(ProblemName p);

	/**
	 * @brief Get the position of the given problem minimum
	 *
	 * @tparam dim the dimension of the space in which the problem is defined
	 * @param p the name of the problem for which the solution position is required
	 * @return RealVector<dim> the position of the minimum
	 */
	template <size_t dim>
	RealVector<dim> get_exact_position(ProblemName p);

};

#include "TestProblems.cpp"