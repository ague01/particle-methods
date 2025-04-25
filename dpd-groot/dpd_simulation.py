import os
import pprint
import numpy as np
import time

# Initialize the Marsenne Twister random number generator
seed = 65647437888080846208636358821643839626
rng = np.random.default_rng(np.random.MT19937(seed))

# Define the particle types constants
P_F = 0
P_W = 1
P_A = 2
P_B = 3


class DPD:
    def __init__(
        self,
        num_particles,
        box_size,
        time_step,
        types: np.ndarray,
        coeffs: list[list[float]],
        bonds: list[tuple[int, int]] | None = None,
        K=None,
        rs=None,
        walls=None,
    ):
        self.num_particles = num_particles
        self.box_size = box_size
        self.cell_size = 1.
        self.dt = time_step
        self.types = types
        self.coeffs = coeffs
        self.bonds = bonds
        self.K = K
        self.rs = rs
        self.walls = walls

        # Initialize random positions
        self.X: np.ndarray = rng.random(size=(num_particles, 2)) * box_size
        # Initialize 0 velocities and forces
        self.V: np.ndarray = np.zeros((num_particles, 2))
        self.F: np.ndarray = np.zeros((num_particles, 2))

        # Initialize cell-list
        self.n_cells = int(self.box_size // self.cell_size)
        self.cells = [[] for _ in range(self.n_cells * self.n_cells)]

        # Initialize walls
        if walls == 'courette':
            # Set type of fluid particles to wall if they are in wall area, i.e. x<1, x>14, should be P_W
            for i in range(num_particles):
                if  self.types[i] == P_F and (self.X[i, 0] < 1. or self.X[i, 0] > 14.):
                    self.types[i] = P_W

    def periodic_displacement(self, pos1, pos2) -> np.ndarray:
        """Compute the displacement vector between two positions accounting for periodic boundaries."""
        delta = pos1 - pos2
        delta = np.where(delta > self.box_size / 2, delta - self.box_size, delta)
        delta = np.where(delta < -self.box_size / 2, delta + self.box_size, delta)
        return delta

    def build_cell_list(self):
        self.cells = [[] for _ in range(self.n_cells * self.n_cells)]
        for idx, pos in enumerate(self.X):
            cell_x = int(pos[0] // self.cell_size)
            cell_y = int(pos[1] // self.cell_size)
            cell_index = cell_y * self.n_cells + cell_x
            self.cells[cell_index].append(idx)

    def compute_forces(self):
        a = 25.
        gamma = 4.5
        sigma = 1.
        isqrt_dt = 1. / np.sqrt(self.dt)
        # Reset forces to zero
        self.F.fill(0)

        # Build cell list
        self.build_cell_list()

        for cell_y in range(self.n_cells):
            for cell_x in range(self.n_cells):
                cell_index = cell_y * self.n_cells + cell_x
                particles = self.cells[cell_index]

                # Check this cell and 8 neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx = (cell_x + dx) % self.n_cells
                        ny = (cell_y + dy) % self.n_cells
                        neighbor_index = ny * self.n_cells + nx
                        neighbors = self.cells[neighbor_index]

                        for i in particles:
                            for j in neighbors:
                                if i >= j:
                                    continue

                                # Calculate distance vector
                                r = self.periodic_displacement(self.X[i], self.X[j])
                                r2 = np.dot(r, r)  # Distance squared norm
                                if r2 < 1.0:
                                    rn = np.sqrt(r2)  # Distance norm
                                    ru = r / rn  # Unit vector

                                    # Calculate F_C
                                    a = self.coeffs[self.types[i]][self.types[j]]
                                    fc = a * (1 - rn) * ru

                                    # Calculate F_R
                                    wr = 1 - rn
                                    xi = rng.normal()
                                    fr = sigma * wr * xi * isqrt_dt * ru

                                    # Calculate F_D
                                    wd = wr ** 2
                                    fd = - gamma * wd * np.dot(ru, self.V[i] - self.V[j]) * ru

                                    # Update forces
                                    self.F[i] += fc + fd + fr
                                    self.F[j] -= fc + fd + fr

        # Compute bonds forces
        if self.bonds is not None:
            assert self.K is not None and self.rs is not None, "K and rs must be provided for bond forces."
            for i, j in self.bonds:
                r = self.periodic_displacement(self.X[i], self.X[j])
                rn = np.linalg.norm(r)
                ru = r / rn
                # Calculate F spring
                fs = self.K * (1 - rn / self.rs) * ru
                # Update forces
                self.F[i] += fs
                self.F[j] -= fs

    def compute_total_momentum(self):
        """Return the total momentum of the system."""
        return np.sum(self.V, axis=0)

    def compute_temperature(self):
        """Return the temperature of the system."""
        return np.mean(self.V ** 2)

    def verlet_step(self):
        """Perform a single Verlet step."""
        # Update positions and enforce periodic boundary conditions
        self.X += self.dt * self.V + 0.5 * (self.dt ** 2) * self.F
        self.X %= self.box_size

        # Update velocities
        V_old = self.V.copy()
        self.V += 0.5 * self.dt * self.F

        # Update forces (V and X are updated to dt+1)
        F_old = self.F.copy()
        self.compute_forces()

        # Update velocities again
        self.V = V_old + 0.5 * self.dt * (F_old + self.F)

    def run_simulation(self, num_steps, save_which=None, save_to=None):
        """Run the simulation for a given number of steps. Save to file snapshots of which particles.

        Parameters
        ----------
        num_steps : int
            Number of steps to run the simulation.
        save_which : list[int] | None (default: None)
            List of indices of particles to save to file. If None, no data is saved.
        save_to : str | None (default: None)
            Path to the file where the data will be saved. If None, no data is saved.
        """

        if save_to or save_which:
            assert save_which is not None, "save_which must be provided if save_to is specified."
            assert save_to is not None, "save_to must be provided if save_which is specified."

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            # Open file and write header
            f = open(save_to, 'w')
            f.write('Time,X,Y,type\n')

        else:
            f = None
            save_which = []

        for step in range(num_steps):
            self.verlet_step()

            # Save positions and types to file for bonded particles
            if f and step % 5 == 0:
                # Save positions and types to file
                for i in save_which:
                    f.write(f"{step * self.dt},{self.X[i, 0]},{self.X[i, 1]},{self.types[i]}\n")
            # Print progress
            if step % 10 == 0:
                print(f'Step {step}/{num_steps} completed.', end='\r')

        if f:
            f.close()
            print(f"Data saved to {save_to}")
        else:
            print("No data saved.")


def step_test(out_path):
    density = 4.0
    box_size = 15.0
    num_particles = int(density * box_size ** 2)
    time_step = 0.01
    max_time = 10.0
    num_steps = int(max_time / time_step)

    types = np.zeros(num_particles, dtype=int)
    coeffs = [[25.,],]

    dpd_simulation = DPD(num_particles, box_size, time_step, types, coeffs)
    print(f'Initial momentum: {dpd_simulation.compute_total_momentum()}')
    print(f'Initial temperature: {dpd_simulation.compute_temperature()}')

    start_time = time.time()
    dpd_simulation.run_simulation(num_steps)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print(f'Final momentum: {dpd_simulation.compute_total_momentum()}')
    print(f'Final temperature: {dpd_simulation.compute_temperature()}')


def step_a(out_path):
    """Study the equilibrium temperature as a function of time step-size."""
    density = 4.0
    box_size = 15.0
    num_particles = int(density * box_size ** 2)
    types = np.zeros(num_particles, dtype=int)
    coeffs = [[25.,],]

    step_sizes = np.linspace(0.001, 0.1, 15)
    equilibrium_temperatures = []

    for time_step in step_sizes:
        time_step = round(time_step, 3)
        print(f"Running simulation with time step: {time_step}")
        num_steps = 1000
        dpd_simulation = DPD(num_particles, box_size, time_step, types, coeffs)
        dpd_simulation.run_simulation(num_steps)
        equilibrium_temperatures.append(dpd_simulation.compute_temperature())

    # Save the results to a CSV file
    csv_path = os.path.join(out_path, 'a_temperature_step_size.csv')
    with open(csv_path, 'w') as f:
        f.write('Time Step Size,Equilibrium Temperature\n')
        for step_size, temperature in zip(step_sizes, equilibrium_temperatures):
            f.write(f'{step_size},{temperature}\n')


def step_b(out_path):
    """Study the motion of particles in Courette flow with chain molecules."""
    density = 4.0
    box_size = 15.0
    num_chains = 42
    num_particles = int(density * box_size ** 2) + num_chains * 7

    # Initialize bonds and types with chain molecules
    bonds = []
    types = np.zeros(num_particles, dtype=int)

    for i in range(num_chains):
        for j in range(6):  # chains of 7 particles
            bonds.append((i * 7 + j, i * 7 + (j + 1)))
            if j < 2:
                types[i * 7 + j] = P_A
            else:
                types[i * 7 + j] = P_B
        # Last type to close the chain
        types[i * 7 + 6] = P_B

    time_step = 0.01
    max_time = 10.0
    num_steps = int(max_time / time_step)

    K = 100.
    rs = 0.1

    coeffs = [
        [50., 25., 25., 200.],
        [25., 1., 300., 200.],
        [25., 300., 25., 200.],
        [200., 200., 200., 0.]
    ]

    out_file = os.path.join(out_path, 'b_courette_flow.csv')

    dpd_simulation = DPD(num_particles, box_size, time_step, bonds=bonds,
                         types=types, coeffs=coeffs, K=K, rs=rs, walls='courette')

    print('Initial momentum:', dpd_simulation.compute_total_momentum())
    print('Initial temperature:', dpd_simulation.compute_temperature())

    dpd_simulation.run_simulation(
        num_steps,
        save_which=range(num_chains * 7),
        save_to=out_file
    )

    print('Final momentum:', dpd_simulation.compute_total_momentum())
    print('Final temperature:', dpd_simulation.compute_temperature())


def main():
    # Create output directory
    out_path = './out/'
    os.makedirs(out_path, exist_ok=True)

    # step_a(out_path)
    # step_test(out_path)
    step_b(out_path)


if __name__ == "__main__":
    main()
