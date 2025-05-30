#pragma once

#include "TestProblems.hpp"

template <>
Problem<2> TestProblems::create_problem(ProblemName p)
{
	if (p == TOWNSEND)
	{
		auto f = [](const RealVector<2> &x)
		{
			double ret = 0;
			ret -= (cos((x[0] - 0.1) * x[1]) * cos((x[0] - 0.1) * x[1]));
			ret -= x[0] * sin(3 * x[0] + x[1]);
			return ret;
		};

		RealVector<2> lb(-10.25, -10.5);
		RealVector<2> ub(10.25, 10.75);

		Problem<2> problem(f, lb, ub);

		problem.add_inequality_constraint([](const RealVector<2> &x)
										  {
			double t = atan2(x[0], x[1]);
			double ret = 0;
			ret += x[0] * x[0] + x[1] * x[1];
			ret -= (2 * cos(t) - cos(2*t)/2 - cos(3*t)/4 - cos(4*t)/8) * (2 * cos(t) - cos(2*t)/2 - cos(3*t)/4 - cos(4*t)/8);
			ret -= (2 * sin(t)) * (2 * sin(t));
			return ret; });

		return problem;
	}
	else if (p == GOMEZ_LEVY)
	{
		auto f = [](const RealVector<2> &x)
		{
			double ret = 0;
			double x2 = x[0] * x[0];
			double y2 = x[1] * x[1];
			ret += 4.0 * x2 - 2.1 * x2 * x2 + (1.0 / 3) * x2 * x2 * x2;
			ret += x[0] * x[1];
			ret += 4 * y2 * y2 - 4 * y2;
			return ret;
		};

		RealVector<2> lb(-4, -4);
		RealVector<2> ub(4, 4);

		Problem<2> problem(f, lb, ub);

		problem.add_inequality_constraint([](const RealVector<2> &x)
										  {
			double ret = 0;
			ret -= std::sin(4 * M_PI * x[0]);
			ret += 2.0 * std::sin(2 * M_PI * x[1]) * std::sin(2 * M_PI * x[1]);
			ret -= 1.5;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<dim> &x)
										  {
			double ret = 0;
			ret += x[0] + x[1];
			ret -= 6.0;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<2> &x)
										  {
			double ret = 0;
			ret -= (x[0] + x[1]);
			ret -= 6.0;
			return ret; });

		return problem;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
Problem<8> TestProblems::create_problem<8>(ProblemName p)
{
	if (p == G10)
	{
		auto f = [](const RealVector<8> &x)
		{
			return x[0] + x[1] + x[2];
		};

		RealVector<8> lb;
		lb << 100, 1000, 1000, 10, 10, 10, 10, 10;
		RealVector<8> ub;
		ub << 1000, 7000, 7000, 1000, 1000, 1000, 1000, 1000;

		Problem<8> problem(f, lb, ub);

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = -1.0;
			ret += 0.0025 * (x[3] + x[5]);
			return ret; });

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = -1.0;
			ret += 0.0025 * (-x[3] + x[4] + x[6]);
			return ret; });

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = -1.0;
			ret += 0.01 * (-x[4] + x[7]);
			return ret; });

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = - x[0] * x[5];
			ret += 833.33252 * x[3] + 100.0 * x[0] - 83333.333;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = - x[1] * x[6];
			ret += 1250.0 * x[4] + x[1] * x[3] - 1250.0 * x[3];
			return ret; });

		problem.add_inequality_constraint([](const RealVector<8> &x)
										  {
			double ret = - x[2] * x[7];
			ret += 1250000.0 + x[2] * x[4] - 2500.0 * x[4];
			return ret; });

		return problem;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
Problem<30> TestProblems::create_problem<30>(ProblemName p)
{
	if (p == GRIEWANK)
	{
		auto f = [](const RealVector<30> &x)
		{
			double sum = 0;
			double prod = 1.0;
			for (int i = 0; i < 30; i++)
			{
				sum += x[i] * x[i];
				prod *= std::cos(x[i] / std::sqrt(double(i) + 1.0));
			}
			return sum / 4000.0 - prod + 1;
		};

		RealVector<30> lb;
		lb << -600, -600, -600, -600, -600, -600, -600, -600, -600, -600,
			-600, -600, -600, -600, -600, -600, -600, -600, -600, -600,
			-600, -600, -600, -600, -600, -600, -600, -600, -600, -600;
		RealVector<30> ub;
		ub << 600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
			600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
			600, 600, 600, 600, 600, 600, 600, 600, 600, 600;

		Problem<30> problem(f, lb, ub);

		return problem;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
Problem<10> TestProblems::create_problem<10>(ProblemName p)
{
	if (p == G7)
	{
		auto f = [](const RealVector<10> &x)
		{
			double ret = 0;
			double a = x[0] * x[0];
			double b = x[1] * x[1];
			double c = x[0] * x[1];
			double d = (x[2] - 10) * (x[2] - 10);
			double e = (x[3] - 5) * (x[3] - 5);
			double f = (x[4] - 3) * (x[4] - 3);
			double g = (x[5] - 1) * (x[5] - 1);
			double h = x[6] * x[6];
			double i = (x[7] - 11) * (x[7] - 11);
			double j = (x[8] - 10) * (x[8] - 10);
			double k = (x[9] - 7) * (x[9] - 7);
			ret += a + b + c - 14 * x[0] - 16 * x[1];
			ret += d + 4 * e + f + 2 * g + 5 * h;
			ret += 7 * i + 2 * j + k + 45;
			return ret;
		};

		RealVector<10> lb;
		lb << -10, -10, -10, -10, -10, -10, -10, -10, -10, -10;
		RealVector<10> ub;
		ub << 10, 10, 10, 10, 10, 10, 10, 10, 10, 10;

		Problem<10> problem(f, lb, ub);

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = -105.0;
			ret += 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7];
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = 3 * (x[0] - 2) * (x[0] - 2);
			ret += 4 * (x[1] - 3) * (x[1] - 3) + 2 * x[2] * x[2] - 7 * x[3] - 120.0;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = 10.0 * x[0];
			ret += - 8 * x[1] - 17 * x[6] + 2 * x[7];
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = x[0] * x[0];
			ret += 2 * (x[1] - 2) * (x[1] - 2) - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5];
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = -8 * x[0];
			ret += 2 * x[1] + 5 * x[8] - 2 * x[9] -12.0;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = 5 * x[0] * x[0];
			ret += 8 * x[1] + (x[2]- 6) * (x[2] - 6) - 2 * x[3] - 40.0;
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = -3 * x[0];
			ret += 6 * x[1] + 12 * (x[8] - 8) * (x[8] - 8) - 7 * x[9];
			return ret; });

		problem.add_inequality_constraint([](const RealVector<10> &x)
										  {
			double ret = 0.5 * (x[0] - 8) * (x[0] - 8);
			ret += 2 * (x[1] - 4) + 3 * x[4] * x[4] - x[5] - 30.0;
			return ret; });

		return problem;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <size_t dim>
double TestProblems::get_exact_value(ProblemName p)
{
	if (p == TOWNSEND && dim == 2)
	{
		return -2.0239883206412607741;
	}
	else if (p == GOMEZ_LEVY && dim == 2)
	{
		return -1.0316284534898774172;
	}
	else if (p == G10 && dim == 8)
	{
		return 7049.248;
	}
	else if (p == G7 && dim == 10)
	{
		return 24.30620;
	}
	else if (p == GRIEWANK && dim == 30)
	{
		return 0.0;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <size_t dim>
RealVector<dim> TestProblems::get_exact_position(ProblemName p)
{
	if (p == TOWNSEND && dim == 2)
	{
		return RealVector<dim>(2.0026010530900202333, 1.1992952337077147362);
	}
	else if (p == GOMEZ_LEVY && dim == 2)
	{
		return RealVector<dim>(0.089842015605160238656, -0.71265640151135767333);
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
RealVector<8> TestProblems::get_exact_position<8>(ProblemName p)
{
	if (p == G10)
	{
		RealVector<8> ret;
		ret << 579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162, 395.5979;
		return ret;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
RealVector<10> TestProblems::get_exact_position<10>(ProblemName p)
{
	if (p == G7)
	{
		RealVector<10> ret;
		ret << 2.1719, 2.3636, 8.7739, 5.0959, 0.9906, 1.4305, 1.3216, 9.8287, 8.2800, 8.3759;
		return ret;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}

template <>
RealVector<30> TestProblems::get_exact_position<30>(ProblemName p)
{
	if (p == GRIEWANK)
	{
		RealVector<30> ret;
		ret << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
		return ret;
	}
	else
		throw std::runtime_error("Problem not implemented.\n \
			Check the documentation for the available problems.\n \
			Check if the dim template parameter is consistent with the specified problem.");
}