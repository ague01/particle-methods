import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Initialize the Marsenne Twister random number generator
seed = 65647437888080846208636358821643839626
rng = np.random.default_rng(np.random.MT19937(seed))

# Set matplotlib backend for non interactive plotting
mpl.use('Agg')


class DPD:
    def __init__(self, num_particles, box_size, time_step):
        self.num_particles = num_particles
        self.box_size = box_size
        self.cell_size = 1.
        self.dt = time_step

        # Initialize random positions
        self.X: np.ndarray = rng.random(size=(num_particles, 2)) * box_size
        # Initialize 0 velocities and forces
        self.V: np.ndarray = np.zeros((num_particles, 2))
        self.F: np.ndarray = np.zeros((num_particles, 2))

        # Initialize cell-list
        self.n_cells = int(self.box_size // self.cell_size)
        self.cells = [[] for _ in range(self.n_cells * self.n_cells)]

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

    def compute_total_momentum(self):
        # Compute the total momentum of the system
        return np.sum(self.V, axis=0)

    def compute_temperature(self):
        """Return the temperature of the system."""

        return np.mean(self.V ** 2)

    def verlet_step(self):
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


    def run_simulation(self, num_steps):
        for step in range(num_steps):
            self.verlet_step()
            if step % 10 == 0:
                print(f'Temperature: {self.compute_temperature()}', end='\r')
                #print(f'Step {step}/{num_steps} completed.', end='\r')


def step_a(out_path):
    density = 4.0
    box_size = 5.0 #TODO: change to 15
    num_particles = int(density * box_size ** 2)
    time_step = 0.01
    max_time = 10.0
    num_steps = int(max_time / time_step)

    dpd_simulation = DPD(num_particles, box_size, time_step)
    print(f'Initial momentum: {dpd_simulation.compute_total_momentum()}')
    print(f'Initial temperature: {dpd_simulation.compute_temperature()}')

    start_time = time.time()
    dpd_simulation.run_simulation(num_steps)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print(f'Final momentum: {dpd_simulation.compute_total_momentum()}')
    print(f'Final temperature: {dpd_simulation.compute_temperature()}')

def main():
    # Create output directory
    out_path = './out/'
    os.makedirs(out_path, exist_ok=True)

    step_a(out_path)


if __name__ == "__main__":
    main()
