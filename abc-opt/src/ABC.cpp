#pragma once

#include "ABC.hpp"
#include <omp.h>
#include <iomanip>
#include "TestProblems.hpp"

template <std::size_t dim>
void ABC<dim>::initialize()
{
	// Instantiate the marsenne twister
	std::random_device rand_dev;
	std::shared_ptr<std::mt19937> generator = std::make_shared<std::mt19937>(rand_dev());
	// Create a shared pointer to the problem
	std::shared_ptr<Problem<dim>> problem = std::make_shared<Problem<dim>>(problem_);

	// Initialize the bees in the colony
	for (std::size_t i = 0; i < colony_size_; ++i)
	{
		// Create and initialize the bee
		colony_.emplace_back(problem, generator, i);
		colony_[i].initialize();
	}

	// Initialize the global best values with the values of the first bee in the colony
	global_best_position_ = colony_[0].get_position();
	global_best_value_ = colony_[0].get_value();
	global_best_constraint_violation_ = colony_[0].get_constraint_violation();

	// Find the best bee in the colony
	for (size_t i = 1; i < colony_size_; ++i)
	{
		if (colony_[i].feasibility_rule(global_best_value_, global_best_constraint_violation_, tol_))
		{
			global_best_position_ = colony_[i].get_position();
			global_best_value_ = colony_[i].get_value();
			global_best_constraint_violation_ = colony_[i].get_constraint_violation();
		}
	}
}


template <std::size_t dim>
void ABC<dim>::optimize(std::ostream& history_out, std::ostream& simulation_out, const int interval)
{
	int current_iter = 0;

	std::uniform_real_distribution<double> distr(0, 1.0);
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	//  Outer optimization loop over all the iterations
	while (current_iter < max_iter_)
	{
		////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////EMPLOYER BEE PHASE///////////////////////////////
		////////////////////////////////////////////////////////////////////////////////

		// Reset the number of feasible bees for the current iteration
		int feasible_bees = 0;

		// Update the position of the bee and prepare data necessary for the onlooker bee phase
		double total_fitness_value = 0;
		double total_constraint_violation = 0;
		for (size_t i = 0; i < colony_size_; ++i)
		{
			// Update the particle
			colony_[i].update_position(MR_, colony_);
			// Prepare the roulette wheel
			total_fitness_value += colony_[i].compute_fitness_value();
			total_constraint_violation += colony_[i].get_constraint_violation();
		}

		////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////ONLOOKER BEE PHASE///////////////////////////////
		////////////////////////////////////////////////////////////////////////////////

		// Compute the probability of each position to be selected by the onlooker bees
		for (size_t i = 0; i < colony_size_; ++i)
		{
			colony_[i].compute_probability(total_fitness_value, total_constraint_violation, tol_);
		}

		// Select, according to the fitness probability, the bees that will be updated
		// In other words the position onlooker bees choose to go to and succesively update.
		unsigned int i = 0, t = 0;
		while (t < colony_size_)
		{
			if (distr(generator) < colony_[i].get_probability())
			{
				t++;
				colony_[i].update_position(MR_, colony_);
			}
			i = (i + 1) % colony_size_;
		}

		////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////SCOUT BEE PHASE//////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////

		// If the number of iterations is a multiple of SPP, reset the bees whose position has not
		// been improved since a time that exceeds the limit value
		if (current_iter % SPP_ == 0)
		{
			size_t max_failure_index = 0;
			for (size_t i = 1; i < colony_size_; ++i)
			{
				if (colony_[i].get_failure_counter() >= colony_[max_failure_index].get_failure_counter())
				{
					max_failure_index = i;
				}
			}

			if (colony_[max_failure_index].get_failure_counter() > limit_)
			{
				colony_[max_failure_index].initialize();
			}
		}

		////////////////////////////////////////////////////////////////////////////////

		for (size_t i = 0; i < colony_size_; ++i)
		{
			// Find the best bee in the colony
			if (colony_[i].feasibility_rule(global_best_value_, global_best_constraint_violation_, tol_))
			{
				global_best_position_ = colony_[i].get_position();
				global_best_value_ = colony_[i].get_value();
				global_best_constraint_violation_ = colony_[i].get_constraint_violation();
			}

			// Count the number of feasible bees
			if (colony_[i].get_constraint_violation() == 0)
			{
				feasible_bees++;
			}
		}

		// Log the optimization process in both history and simulation streams.
		// NB: Branch prediction will optimize well the following branch since it has a constant condition
		if (log_verbose_ && current_iter % interval == 0)
		{
			// Print Iters,value,violation,threshold,feasible_particles to the history
			// NB: threshold is 0 for compatibility con the other algorithms
			history_out << current_iter << ","
						<< global_best_value_ << ","
						<< global_best_constraint_violation_ << ","
						<< feasible_bees <<  std::endl;
			// Print all the particles position and flag the best one to the simulation stream
			for (int i = 0; i < colony_size_; i++)
			{
				simulation_out << current_iter << ",";
				for (int d = 0; d < dim; d++)
					simulation_out << colony_[i].get_position()[d] << ",";
				simulation_out << ((colony_[i].get_value() == global_best_value_) ? 1 : 0) << std::endl;
			}
		}

		// Update the current iteration
		current_iter++;
	}
}

template <std::size_t dim>
void ABC<dim>::initialize_parallel()
{
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Workload distribution among MPI processes:
	//	- Each process will have a number of bees equal to ceil(colony_size_ / mpi_size)
	//	- If the number of bees is not divisible by the number of processes, the first process will have less bees
	unsigned int process_colony_size = colony_size_ / mpi_size;
	if (colony_size_ % mpi_size != 0)
	{
		process_colony_size++;
		if (mpi_rank == 0)
		{
			process_colony_size = colony_size_ - process_colony_size * (mpi_size - 1);
		}
	}
	// Each process needs to know only its colony
	colony_size_ = process_colony_size;

	// Create a shared pointer to the problem
	std::shared_ptr<Problem<dim>> problem = std::make_shared<Problem<dim>>(problem_);

	// Resize the colony, needed since threads performs direct access to the vector
	colony_.resize(colony_size_);

#pragma omp parallel shared(colony_, problem, global_best_position_, global_best_value_, global_best_constraint_violation_)
	{

		// Instantiate the marsenne twister, one for each thread since it is not thread safe
		std::random_device rand_dev;
		std::shared_ptr<std::mt19937> generator = std::make_shared<std::mt19937>(rand_dev());

#pragma omp single
		{
			colony_[0] = Bee(problem, generator, 0);
			colony_[0].initialize();

			// Initialize the global best values
			global_best_position_ = colony_[0].get_position();
			global_best_value_ = colony_[0].get_value();
			global_best_constraint_violation_ = colony_[0].get_constraint_violation();
		}

// Initialize the bees in the colony
#pragma omp for schedule(static)
		for (std::size_t i = 1; i < colony_size_; ++i)
		{
			// Create and initialize the bee
			colony_[i] = Bee(problem, generator, i);
			colony_[i].initialize();
		}

		// Initialize the thread local best index
		size_t local_best_index = 0;

// Find the local best bee for each thread
#pragma omp for nowait schedule(static)
		for (size_t i = 1; i < colony_size_; ++i)
		{
			if (colony_[i].feasibility_rule(colony_[local_best_index].get_value(), colony_[local_best_index].get_constraint_violation(), tol_))
			{
				local_best_index = i;
			}
		}

// Reduction: Each thread updates the global values if it has a better solution
#pragma omp critical(update_global_best_init)
		if (colony_[local_best_index].feasibility_rule(global_best_value_, global_best_constraint_violation_, tol_))
		{
			global_best_position_ = colony_[local_best_index].get_position();
			global_best_value_ = colony_[local_best_index].get_value();
			global_best_constraint_violation_ = colony_[local_best_index].get_constraint_violation();
		}
	}
}

template <std::size_t dim>
void ABC<dim>::optimize_parallel(std::ostream& history_out, std::ostream& simulation_out, const int interval)
{
	int current_iter = 0;
	size_t local_best_index = 0;

//  Outer optimization loop over all the iterations
#pragma omp parallel shared(colony_) firstprivate(local_best_index, current_iter)
	{
		double local_fitness_value;
		double local_constraint_violation;

		// Instantiate the marsenne twister since it is not thread safe
		std::uniform_real_distribution<double> distr(0, 1.0);
		std::random_device rand_dev;
		std::mt19937 generator(rand_dev());

		// Divide the colony among the threads, each thread has its own private colony, avoiding false sharing issues
		// observed whith threads directly accessing different elements of the same vector
		size_t num_private_colony = colony_size_ / omp_get_num_threads();
		size_t thread_id = omp_get_thread_num();
		typename std::vector<Bee<dim>>::iterator first = colony_.begin() + num_private_colony * thread_id;
		typename std::vector<Bee<dim>>::iterator last = (thread_id == omp_get_num_threads() - 1) ? colony_.end() : colony_.begin() + num_private_colony * (thread_id + 1);
		std::vector<Bee<dim>> private_colony(first, last);

		while (current_iter < max_iter_)
		{
			////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////EMPLOYER BEE PHASE///////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			local_fitness_value = 0;
			local_constraint_violation = 0;

			// Update the position of the bee and prepare data necessary for the onlooker bee phase
			for (size_t i = 0; i < private_colony.size(); ++i)
			{
				// Update the particle
				private_colony[i].update_position(MR_, private_colony);
				// Prepare the roulette wheel
				local_fitness_value += private_colony[i].compute_fitness_value();
				local_constraint_violation += private_colony[i].get_constraint_violation();
			}

			////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////ONLOOKER BEE PHASE///////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// Compute the probability of each position to be selected by the onlooker bees
			for (size_t i = 0; i < private_colony.size(); ++i)
			{
				private_colony[i].compute_probability(local_fitness_value, local_constraint_violation, tol_);
			}

			// Select, according to the fitness probability, the bees that will be updated
			// In other words the position onlooker bees choose to go to and succesively update.
			unsigned int i = 0, t = 0;
			while (t < private_colony.size())
			{
				if (distr(generator) < private_colony[i].get_probability())
				{
					t++;
					private_colony[i].update_position(MR_, private_colony);
				}
				i = (i + 1) % private_colony.size();
			}

			////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////SCOUT BEE PHASE//////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// If the number of iterations is a multiple of SPP, reset the bees whose position has not
			// been improved since a time that exceeds the limit value
			if (current_iter % SPP_ == 0)
			{
				size_t max_failure_index = 0;
				for (size_t i = 1; i < private_colony.size(); ++i)
				{
					if (private_colony[i].get_failure_counter() >= private_colony[max_failure_index].get_failure_counter())
					{
						max_failure_index = i;
					}
				}

				if (private_colony[max_failure_index].get_failure_counter() > limit_)
				{
					private_colony[max_failure_index].initialize();
				}
			}

			////////////////////////////////////////////////////////////////////////////////

			// Find the best bee in the private colony
			for (size_t i = 0; i < private_colony.size(); ++i)
			{
				if (private_colony[i].feasibility_rule(private_colony[local_best_index].get_value(), private_colony[local_best_index].get_constraint_violation(), tol_))
				{
					local_best_index = i;
				}
			}

			// Update the current iteration
			current_iter++;
		}

// Reduction: each thread updates the global values (locally to each process) if it has a better solution
#pragma omp critical(update_global_best_init)
		if (private_colony[local_best_index].feasibility_rule(global_best_value_, global_best_constraint_violation_, tol_))
		{
			global_best_position_ = private_colony[local_best_index].get_position();
			global_best_value_ = private_colony[local_best_index].get_value();
			global_best_constraint_violation_ = private_colony[local_best_index].get_constraint_violation();
		}

		// file_out << "Best value: " << global_best_value_ << "       Violation: "<< global_best_constraint_violation_<< std::endl;
	}

	// Create the custom MPI reduction operation
	MPI_Op mpi_custom_reduction;
	MPI_Op_create((MPI_User_function *)ABC<dim>::custom_reduction, 1, &mpi_custom_reduction);

	// Create the mpi type for the ABC_result structure
	MPI_Datatype mpi_abc_result;
	MPI_Type_contiguous(2 + dim, MPI_DOUBLE, &mpi_abc_result);
	MPI_Type_commit(&mpi_abc_result);

	int mpi_rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	// Initialize the data structure for the local process to be reduced
	double loc_result[2 + dim];
	double glob_result[2 + dim];

	loc_result[0] = global_best_value_;
	loc_result[1] = global_best_constraint_violation_;
	for (std::size_t i = 0; i < dim; ++i)
		loc_result[2 + i] = global_best_position_[i];

	// MPI reduction over the process global best values to find the global global best values
	MPI_Reduce(&loc_result, &glob_result, 1, mpi_abc_result, mpi_custom_reduction, 0, MPI_COMM_WORLD);
	if (mpi_rank == 0)
	{
		global_best_value_ = glob_result[0];
		global_best_constraint_violation_ = glob_result[1];
		for (std::size_t i = 0; i < dim; ++i)
			global_best_position_[i] = glob_result[2 + i];
	}
}

template <std::size_t dim>
void ABC<dim>::custom_reduction(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
{
	double *in = (double *)invec;
	double *inout = (double *)inoutvec;

	bool is_better = false;
	// Apply the feasibility rule to check wheter the invec solution is better than inout
	if (in[1] == 0 && inout[1] > 0)
		is_better = true;
	else if (in[1] > 0 && inout[1] == 0)
		is_better = false;
	else if (in[1] == 0 && inout[1] == 0)
		is_better = in[0] < inout[0];
	else
		is_better = in[1] < inout[1];

	if (is_better)
	{
		inout[0] = in[0];
		inout[1] = in[1];
		for (std::size_t i = 0; i < dim; ++i)
			inout[i + 2] = in[i + 2];
	}
}

template <std::size_t dim>
void ABC<dim>::print_results(std::ostream &out) const
{
	out << std::setprecision(20) << "Best value found: " << global_best_value_ << std::endl;
	out << "Best position found: (";
	for (std::size_t i = 0; i < dim; ++i)
		out << global_best_position_[i] << ", ";
	out << "\b\b)" << std::endl;
	out << "Total constraint violation obtained: " << global_best_constraint_violation_ << std::endl;
}