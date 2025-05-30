#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "ABC.hpp"
#include "TestProblems.hpp"

using namespace type_traits;
namespace fs = std::filesystem;

/*#define dimension 2
#define test_problem TestProblems::GOMEZ_LEVY
#define problem_name "GOMEZ_LEVY"*/

/*#define dimension 2
#define test_problem TestProblems::TOWNSEND
#define problem_name "TOWNSEND"*/

/*#define dimension 8
#define test_problem TestProblems::G10
#define problem_name "G10"*/

/*#define dimension 30
#define test_problem TestProblems::GRIEWANK
#define problem_name "GRIEWANK"*/

#define dimension 10
#define test_problem TestProblems::G7
#define problem_name "G7"

/**
 * @brief Optimize the given problem using the specified algorithm and log the results to the output files
 * Logging is made both for history and simulation data
 * @param log_verbose if true, the optimization process is logged to the output files (default: false)
 * @return int 0 if the optimization is successful, -1 otherwise
 */
int
optimize ()
{
  constexpr int log_interval = 2;
  constexpr bool log_verbose = false;

  int iter = 8000;
  int particles = 80;
  auto problem = TestProblems::create_problem<dimension> (test_problem);

  // Preliminary informations to std out
  std::cout << "Serial optimization with ABC" << std::endl;
  std::cout << "Logs in /output/abc_optimize.csv" << std::endl;
  std::cout << "Simulation data in /output/abc_simulation.csv" << std::endl;

  std::cout << "Problem: " << problem_name << std::endl;
  std::cout << "Max iterations: " << iter << std::endl;
  std::cout << "Num particles: " << particles << std::endl;

  std::ofstream history_out;
  history_out.open ("../output/optimize_abc_" + std::to_string (1) + ".csv");
  if (!history_out)
    {
      std::cout << "Error opening file" << std::endl;
      return -1;
    }
  std::ofstream simulation_out;
  simulation_out.open ("../output/simulation_abc_" + std::to_string (1) + ".csv");
  if (!simulation_out)
    {
      std::cout << "Error opening file" << std::endl;
      return -1;
    }

  std::vector<double> results (20);

  // Repeat the optimization 20 times to get a better average
  for (int i = 0; i < 20; i++)
    {
      ABC<dimension> opt = ABC<dimension> (problem, particles, iter);
      opt.set_log_verbose (log_verbose);

      // Write comments and header
      history_out << "# Fitness, constraints violation and feasible particles over iterations" << std::endl;
      history_out << "# Dimension: " << dimension << std::endl;
      history_out << "# Particles: " << particles << std::endl;
      history_out << "# Problem: " << problem_name << std::endl;
      history_out << "iters,value,violation,feasible_particles" << std::endl;

      simulation_out
          << "# Data of all the particles position at every interval iterations, the best one is flagged"
          << std::endl;
      simulation_out << "# Dimension: " << dimension << std::endl;
      simulation_out << "# Particles: " << particles << std::endl;
      simulation_out << "# Problem: " << problem_name << std::endl;
      simulation_out << "iter,";
      for (int i = 0; i < dimension; i++)
        simulation_out << "x" << i << ",";
      simulation_out << "isbest" << std::endl;

      // Optimize the problem storing data to files
      opt.initialize ();
      opt.optimize (history_out, simulation_out, log_interval);

      // Get the exact global minimum
      double exact_value = TestProblems::get_exact_value<dimension> (test_problem);

      // Print the final data
      opt.print_results ();
      std::cout << "Exact value: " << TestProblems::get_exact_value<dimension> (test_problem) << std::endl;
      std::cout << "Absolute error: " << std::abs (opt.get_global_best_value () - exact_value) << std::endl;
      std::cout << "Relative error: " << std::abs ((opt.get_global_best_value () - exact_value) / exact_value)
                << std::endl;
      std::cout << "Absolute distance: "
                << (TestProblems::get_exact_position<dimension> (test_problem)
                    - opt.get_global_best_position ())
                       .norm ()
                << std::endl;
      if (log_verbose)
        break;
      // Store the results in the vector
      results[i] = opt.get_global_best_value ();
    }
  // Close file streams
  history_out.close ();
  simulation_out.close ();

  // Print the average results
  std::cout << std::endl;
  std::cout << "Average results over 10 iterations:" << std::endl;
  std::cout << "Best value: " << *std::min_element (results.begin (), results.end ()) << std::endl;
  std::cout << "Worst value: " << *std::max_element (results.begin (), results.end ()) << std::endl;
  std::cout << "Average value: " << std::accumulate (results.begin (), results.end (), 0.0) / results.size ()
            << std::endl;
  std::cout << "Standard deviation: "
            << std::sqrt (std::accumulate (results.begin (), results.end (), 0.0,
                                       [](double sum, double val) { return sum + val * val; }) /
                                       results.size () -
                                   std::pow (std::accumulate (results.begin (), results.end (), 0.0) / results.size (),
                                             2))
            << std::endl;
  return 0;
}

// test the time as function of the number of particles
int
time_numparticles_test ()
{
  int iter = 10000;
  int max_particles = 2048;
  constexpr int log_multiplier = 2;
  constexpr int init_particles = 32;
  auto problem = TestProblems::create_problem<dimension> (test_problem);

  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

  // Get the number of omp threads
  int tot_threads = 0;
#pragma omp parallel
  {
#pragma omp single
    tot_threads = omp_get_num_threads ();
  }

  std::ofstream file_out;
  if (mpi_rank == 0)
    {
      tot_threads *= mpi_size;
      // Preliminary informations to std out
      std::cout << "Time and Speedup as function of colony size" << std::endl;
      std::cout << "Logs in /output/time_numparticles_abc_" + std::to_string (tot_threads) + ".csv"
                << std::endl;

      std::cout << "Problem: " << problem_name << std::endl;
      std::cout << "Max iterations: " << iter << std::endl;
      std::cout << "Log multiplier: " << log_multiplier << std::endl;

      // Initialize the file
      file_out.open ("../output/time_numparticles_abc_" + std::to_string (tot_threads) + ".csv");
      if (!file_out)
        {
          std::cout << "Error opening file" << std::endl;
        }
      // Write comments and header to file
      file_out << "# Execution time as function of the colony size" << std::endl;
      file_out << "# Problem: " << problem_name << std::endl;
      file_out << "# Dimension: " << dimension << std::endl;
      file_out << "# Threads: " << tot_threads << std::endl;
      file_out << "Num_particles,Serial_time,Parallel_time,Speedup" << std::endl;

      std::cout << "Starting test from 1 to " << max_particles << " particles" << std::endl;
      std::cout << "Logging every x" << log_multiplier << " iterations" << std::endl;
    }
  for (int i = init_particles; i <= max_particles; i *= log_multiplier)
    {
      std::chrono::time_point<std::chrono::system_clock> t1, t2, t3, t4;
      if (mpi_rank == 0)
        {
          // Print progress to stdout
          std::cout << "Starting test with " << i << " particle(s)" << std::endl;
          // Optimize the problem serially
          ABC<dimension> opt = ABC<dimension> (problem, i, iter);
          opt.set_log_verbose (false);

          opt.initialize ();
          t1 = std::chrono::high_resolution_clock::now ();
          opt.optimize ();
          t2 = std::chrono::high_resolution_clock::now ();
        }
      // Optimize parallel
      ABC<dimension> opt = ABC<dimension> (problem, i, iter);
      opt.set_log_verbose (false);

      opt.initialize_parallel ();
      t3 = std::chrono::high_resolution_clock::now ();
      opt.optimize_parallel ();
      t4 = std::chrono::high_resolution_clock::now ();
      if (mpi_rank == 0)
        {
          // Write data to file
          file_out << i << ",";
          file_out << std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1).count () << ",";
          file_out << std::chrono::duration_cast<std::chrono::milliseconds> (t4 - t3).count () << ",";
          file_out << double (std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1).count ())
                          / std::chrono::duration_cast<std::chrono::milliseconds> (t4 - t3).count ()
                   << std::endl;
        }
    }
  if (mpi_rank == 0)
    {
      // Close the file
      file_out.close ();
    }

  return 0;
}

int
serial_parallel_test ()
{
  // Initialize problem and solver parameters
  constexpr int log_interval = 50;
  int iter = 6000;
  int particles = 300;
  auto problem = TestProblems::create_problem<dimension> (test_problem);

  int mpi_rank = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

  ABC<dimension> opt = ABC<dimension> (problem, particles, iter);
  double time_serial = 0.0;
  // Preliminary informations and serial optimization
  if (mpi_rank == 0)
    {
      std::cout << "Serial vs Parallel optimization test with ABC" << std::endl;
      std::cout << "Problem: " << problem_name << std::endl;
      std::cout << "Max iterations: " << iter << std::endl;
      std::cout << "Num particles: " << particles << std::endl;

      // Test the serial version
      std::cout << std::endl;
      std::cout << "--- Serial optimizer testing ---" << std::endl;

      auto t1 = std::chrono::high_resolution_clock::now ();
      opt.initialize ();
      opt.optimize ();
      auto t2 = std::chrono::high_resolution_clock::now ();
      opt.print_results ();

      float exact_value = TestProblems::get_exact_value<dimension> (test_problem);

      std::cout << "Exact value: " << exact_value << std::endl;
      std::cout << std::setprecision (20) << "Absolute error: " << opt.get_global_best_value () - exact_value
                << std::endl;
      std::cout << std::setprecision (20)
                << "Relative error: " << (opt.get_global_best_value () - exact_value) / exact_value
                << std::endl;
      std::cout << std::setprecision (20) << "Absolute distance: "
                << (TestProblems::get_exact_position<dimension> (test_problem)
                    - opt.get_global_best_position ())
                       .norm ()
                << std::endl;
      time_serial = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1).count ();
      std::cout << std::setprecision (20) << "Elapsed initialization + optimization time: " << time_serial
                << std::endl;
    }
  std::chrono::time_point<std::chrono::system_clock> t1, t2;

  // Test the parallel version
  if (mpi_rank == 0)
    {
      std::cout << std::endl;
      std::cout << "--- Parallel optimizer testing ---" << std::endl;
    }
  ABC<dimension> opt_p = ABC<dimension> (problem, particles, iter);
  if (mpi_rank == 0)
    {
      t1 = std::chrono::high_resolution_clock::now ();
    }
  opt.initialize_parallel ();
  opt.optimize_parallel ();
  if (mpi_rank == 0)
    {
      t2 = std::chrono::high_resolution_clock::now ();
      float exact_value = TestProblems::get_exact_value<dimension> (test_problem);
      opt.print_results ();
      std::cout << std::setprecision (20) << "Absolute error: " << opt.get_global_best_value () - exact_value
                << std::endl;
      std::cout << std::setprecision (20)
                << "Relative error: " << (opt.get_global_best_value () - exact_value) / exact_value
                << std::endl;
      std::cout << std::setprecision (20) << "Absolute distance: "
                << (TestProblems::get_exact_position<dimension> (test_problem)
                    - opt.get_global_best_position ())
                       .norm ()
                << std::endl;
      double time_parallel = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1).count ();
      std::cout << std::setprecision (20) << "Elapsed initialization + optimization time: " << time_parallel
                << std::endl;
      std::cout << "Speedup: " << time_serial / time_parallel << std::endl;
    }

  return 0;
}

int
main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0)
    {
      std::cout << "Running with " << mpi_size << " MPI processes" << std::endl;
    }

  // Check the number of arguments
  if (argc != 2)
    {
      if (mpi_rank == 0)
        {
          std::cout << "Usage: ./test [optimize | serial_parallel | time_numparticles]" << std::endl;
        }
      MPI_Finalize ();
      return -1;
    }
  // Create if it not exist the output directory
  if (mpi_rank == 0)
    {
      fs::create_directory ("../output");
    }
  // Get from command line the required test
  std::string test = argv[1];
  if (test == "optimize")
    optimize ();
  else if (test == "serial_parallel")
    serial_parallel_test ();
  else if (test == "time_numparticles")
    time_numparticles_test ();
  else if (mpi_rank == 0)
    {
      std::cout << "Error: unknown test. Run without arguments to display available test." << std::endl;
    }

  MPI_Finalize ();

  return 0;
}