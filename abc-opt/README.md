# ABC - Artificial Bee Colony Optimization

The ABC algorithm is inspired by the foraging behavior of honey bees. It simulates the intelligent food searching behavior of a honey bee swarm. Constrained optimization is available with a constraint handling technique based on the **Deb's feasibility-based rule**. Thread-level OpenMP and Process-level MPI parallelization is also available.

# Interface

The `Problem` class describes an optimization problem to be optimized. The user needs to instantiate a `Problem` object and set the space dimension in which it belongs, its fitness function, constraits, and bounds. It provides the following interface to build a problem:

- `Constructor` to provide: fitness function as `std::function`, lower bounds and upper bounds both as `Eigen::Matrix<double, dim, 1>` where `dim` is the search space dimension.
- `add_equality_constraint` to add an equality constraint $f$ as `std::function` (or any cast compatible type) s.t. $f(\vec{x})=0$
- `add_inequality_constraint` to add an inequality constraint $f$ as `std::function` (or any cast compatible type) s.t. $f(\vec{x})\le0$

The `ABC` class describes the interface for the ABC optimizer implementation. The user needs to instantiate an object of the optimizer providing the `Problem` as parameter.

The following code implements a possible basic usage. Firstly the problem to be optimized must be defined providing the fitness function `f`, the bounds and any constraint

```cpp
auto f = [](const RealVector<2> &x)
{
  return x[0] * x[0] + x[1] * x[1];
};
RealVector<2> lb(-4, -4);
RealVector<2> ub(4, 4);
Problem<2> problem(f, lb, ub);
problem.add_inequality_constraint([](const RealVector<2> &x) {
  return - std::sin(4 * M_PI * x[0]);
});
```

Then the ABC solver must be created providing the problem, and then it can be initialized and solved.

```cpp
ABC<2> opt = ABC<2>(problem, 100, 1000);
opt.initialize();
opt.optimize();
opt.print_results();
```

Here the solver has been initialized for solving a 2 dimensional problem `problem` with 100 particles and 1000 iterations.

## Requirements

- CMake
- C++17
- OpenMP 3.1
- MPI 3.0
- Python3
- Eigen 3.3

## Compile and Run

1. Create the folders needed for building the project and saving the output `.csv` and `.png` files

   ```bash
   mkdir build
   mkdir output
   ```

2. Move into the build folder

   ```bash
   cd ./build
   ```

3. Execute cmake (it may require the flag with the explicit path to the Eigen library if it isn't in the default folder)

   ```bash
   cmake ..
   ```

4. Build the project

   ```bash
   cmake --build .
   ```

5. Launch the test executable. Suitable test name must be provided (a complete list of the available test is shown if no parameter is provided)

   ```bash
   ./test [test_name]
   ```

   The execution of a test may produce an output file `test-name_algorithm-name_num-threads.csv` that can be found under the `output` folder.
6. Plot the results in a graphical way using the scripts in the `/script` folder. `csv_plotter.csv` takes as argument only the filename of one .csv file stored in the the `output` folder and plots only data related to a single run of a single solver. While `csv_scaling.py` is used to plot data from many files and takes as argument a type of plot among `speedup` for the parallel speedup and `strongsingle` for the strong scalability of each single solver. Lastly the `csv_simulation.py` plots the evolution of the swarm on a Gomez-Levy 2D problem.
   It does requires the filename in which the simulation data for the GL problem has been stored, i.e. `simulation_abc_xx.csv` (it may takes a while depending on the number of iterations).

   ```bash
   cd ../scripts
   python csv_plotter.py test-name_abc_threads-num.csv
   python csv_scaling.py [strongsingle | speedup]
   python csv_animation.py simulation_abc_1.csv
   ```

   All the plots are stored in the `/plots` folder.
   NB: we choose to have different scripts since each of them manages data in a different way: the first takes a single file, the second aggregates more files data, and the last manages files to produce an animation.

## ABC algorithm description

The Artificial Bee Colony (ABC) algorithm is a metaheuristic optimization technique inspired by the foraging behavior of honey bees. It was introduced by Karaboga and Basturk and has proven effective for solving complex optimization problems, including constrained optimization problems (COPs).

The version developed in this project does refer to the Karaboga's ABC implementation for solving constrained optimization problem, which has been proposed [here](https://www.sciencedirect.com/science/article/pii/S1568494610003066).

The ABC algorithm simulates the intelligent foraging behavior of a honey bee swarm. The algorithm consists of three types of bees: employed bees, onlookers, and scouts, which work together to find the optimal solution.

## ABC key Features

- `Exploration and Exploitation`: The ABC algorithm balances exploration (global search) and exploitation (local search) through the cooperative behavior of the bees.
- `Adaptivity`: It dynamically adapts the search process based on the feedback from the environment, improving its ability to find optimal solutions.
- `Flexibility`: ABC can be applied to various types of optimization problems without significant modifications.

## Algorithm Description

It is worth to explain that, although different kind of bees are presented, they don't have to be considered effectively as if they are objects with different behaviours. In fact, in the algorithm, the employer phase, the scout phase and so on, just represent and characterize the different phases of the optimization single step.

The ABC algorithm involves the following steps:

- `Initialization`: Generate an initial population of food sources (solutions) randomly.
- `Employed Bee phase`: Each employed bee evaluates the fitness of its food source and explores neighboring solutions, updating its position with a probability related to the algorithm parameter MR:

$$
x_{ij} = \begin{cases}
x_{ij} + \beta_{ij}(x_{ij} - x_{kj}), & \text{if } R_j < MR \\
x_{ij}, & \text{otherwise}.
\end{cases}$$


- `Onlooker Bee Phase`: Onlooker bees select new food sources to be followed, based on their probability $p_i$, which is proportional to the fitness of the food source.

$$
p_i = \begin{cases}
  0.5 + \frac{fitness_i}{\sum\limits_{j=1}^{n} fitness_j} \times 0.5 & \text{if feasible} \\
  \left( 1 - \frac{violation_i}{\sum\limits_{j=1}^{n} (violation_j)} \right) \times 0.5 & \text{otherwise}
\end{cases}
$$

- `Scout Bee Phase`: If a food source cannot be improved further, it is abandoned, and a scout bee randomly searches for a new food source, meaning that the bee is reinitialized.
- `Termination`: The process repeats until a stopping criterion (maximum number of iterations or acceptable solution quality) is met.

## Constrained Optimization

The ABC algorithm handles constrained optimization problems by incorporating a penalty function approach. The feasibility-based rule ensures that:
- feasible solutions are preferred over infeasible ones
- among feasible solutions, the one with better objective function value is chosen
- among infeasible solutions, the one with the smallest total constraint violation is preferred.

## Code structure

Two main classes have been developed:

-  `Bee` Represents a single bee in the swarm and manages its initialization, evaluation, and update at each iteration. It implements the feasibility rule and the search behavior of employed, onlooker, and scout bees.
-  `ABC` wraps the colony of bees and the ABC algorithm, providing methods for initialization, execution, and output. The class supports both serial and parallel implementations

The `ABC` class provides methods for the **serial** and the **parallel** implementation for both initialization and optimization methods. The parallel implementation uses OpenMP for multithreading and MPI for multiprocessing.

In the parallel version of the ABC algorithm, the entire colony of bees is divided equally among the available processors. This parallel implementation is designed for shared memory architectures and has been developed according to what proposed [here](https://ieeexplore.ieee.org/document/5393726), in fact some modifications with respect to the classical algorithm are needed in order to parallelize it.
In particular, each thread handles a subset of the total bee's colony and operate completely on it, neglecting what is done by the other threads. This rearrangement of the logic of the classical version ABC algorithm, allows not to have significant overhead due to synchronization in the parallel implementation.
In fact, just a final reduction at the end of the optimize process is needed. It allows to select the best bee among all the local colonies and will determine the global result.
The MPI parallelization has been performed exploiting the same algorithmic idea. The global colony is divided among different MPI processes, each one operating independently from the other one, and a custom MPI reduction based on the feasibility rule has been implemented in order to find, at the end of the optimize process, the global optimal solution.

## Conclusion

The ABC algorithm is a robust optimization method suitable for a wide range of optimization problems, including constrained ones. Its parallel implementation ensures scalability and efficiency, making it a valuable tool for solving complex problems. Other details about the implementation, the usage and the tests results can be found in the report PDF file.