#pragma once

#include "Problem.hpp"
#include "Bee.hpp"
#include "mpi.h"

/**
 * @brief Specialization of the Optimizer class implementing the Artificial Bee Colony Optimization (ABC)
 * algorithm. All the information regarding the analysis and the theory behind this algorithm can be found in
 * the following paper: https://link.springer.com/content/pdf/10.1007/978-3-540-72950-1_77.pdf
 *
 * @param colony_size the number of particles in the swarm
 * @param max_iter the maximum number of iterations
 * @param tol the tolerance used for checking constraint conditions
 * @tparam dim the dimension of the space in which the function is defined
 */
template <std::size_t dim> class ABC
{
private:
  Problem<dim> problem_;
  int colony_size_;
  int max_iter_;
  int limit_;
  double MR_;
  int SPP_;
  bool log_verbose_;

  RealVector<dim> global_best_position_;
  double global_best_value_;
  double global_best_constraint_violation_;

  std::vector<Bee<dim>> colony_;

  double violation_threshold_;
  double tol_;

public:
  /**
   * @brief Construct a new ABC optimizator object for the given problem.
   *
   * @param problem the Problem to be optimized
   * @param colony_size the number of particles in the swarm
   * @param max_iter the maximum number of iterations
   * @param limit the maximum number of iterations in which a Bee has not improved (default is 0.5 *
   * colony_size * dim)
   * @param SPP the number of iterations in which the scout bee phase is performed (default is 0.5 *
   * colony_size * dim)
   * @param MR the modification rate
   * @param tol the tolerance used for checking constraint conditions
   * @param log_verbose if true, the optimization process is logged to the output files (default: false)
   */

  ABC (const Problem<dim> &problem, int colony_size = 50, int max_iter = 7000, int limit = -1, int SPP = -1,
       double MR = 0.8, double tol = 1e-6, bool log_verbose = false)
      : problem_ (problem), colony_size_ (colony_size), max_iter_ (max_iter), MR_ (MR), tol_ (tol),
        limit_ (limit == -1 ? static_cast<int> (colony_size * dim * 0.5) : limit),
        SPP_ (SPP == -1 ? static_cast<int> (colony_size * dim * 0.5) : SPP), log_verbose_ (log_verbose)
  {
  }

  ~ABC () = default;

  /**
   * @brief Initialize the optimizer to start the optimization process
   * This method initializes the swarm of particles and all the parameters needed by the algorithm
   */
  void initialize ();

  /**
   * @brief Initialize the optimizer to start the optimization process using OMP parallel constructs
   * This initialization must be used if the optimization will be performed using optimize_parallel method
   */
  void initialize_parallel ();

  /**
   * @brief Optimize the objective function and print to the given streams the history of the optimization
   * process if log_verbose is true
   *
   * @param optimum_history the stream to which print the history as: iteration, fitness,
   * constraint_violation, feasible_particles
   * @param simulation_history the stream to which print the data to produce a simulation of the optimization
   * process
   * @param interval the interval in number of iterations to print the history
   */
  void optimize (std::ostream &history_out = std::cout, std::ostream &simulation_out = std::cout,
                 const int interval = 50);

  /**
   * @brief Optimize the objective function in parallel and print to the given streams the history of the
   * optimization process if log_verbose is true The parallelism type is MPI multi-processing with OMP
   * multi-threading
   *
   * @param optimum_history the stream to which print the history as: iteration, fitness,
   * constraint_violation, feasible_particles
   * @param simulation_history the stream to which print the data to produce a simulation of the optimization
   * process
   * @param interval the interval in number of iterations to print the history
   */
  void optimize_parallel (std::ostream &history_out = std::cout, std::ostream &simulation_out = std::cout,
                          const int interval = 50);

  /**
   * @brief Print the results of the optimization process to the given output stream
   *
   * @param out output stream where to print results
   */
  void print_results (std::ostream &out = std::cout) const;

  /**
   * @brief Get the actual global best value found by the algorithm
   *
   * @return double the global best value
   */
  double
  get_global_best_value () const
  {
    return global_best_value_;
  }

  /**
   * @brief Get the position of the global best minimum found by the algorithm
   *
   * @return const RealVector<dim>& a const reference to the global best position vector
   */
  const RealVector<dim> &
  get_global_best_position () const
  {
    return global_best_position_;
  }

  double
  get_global_best_constraint_violation () const
  {
    return global_best_constraint_violation_;
  }

  /**
   * @brief Custom MPI reduction function implementing the feasibility-based rule for the ABC algorithm
   * This function must be used for the MPI_Op_create function to create a custom MPI_Op for the reduction
   *
   * @param invec a pointer to the first input array (or value) to be reduced
   * @param inoutvec a pointer to the second input array (or value) of the reduction and where the output is
   * stored
   * @param len the number of elements in the input arrays (or 1 if the input is a single value)
   * @param datatype the MPI datatype of the input arrays
   */
  static void custom_reduction (void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

  /**
   * @brief Set the verbosity for logging purposes
   *
   * @param log_verbose the verbosity for logging purposes
   */
  void
  set_log_verbose (bool log_verbose)
  {
    log_verbose_ = log_verbose;
  }
};

#include "ABC.cpp"