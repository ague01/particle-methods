from ast import main
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
        self.dt = time_step
        # Initialize random positions
        self.X: np.ndarray = rng.random(size=(num_particles, 2)) * box_size
        # Initialize 0 velocities and forces
        self.V: np.ndarray = np.zeros((num_particles, 2))
        self.F: np.ndarray = np.zeros((num_particles, 2))

    def periodic_displacement(self, pos1, pos2) -> np.ndarray:
        """Compute the displacement vector between two positions accounting for periodic boundaries."""
        delta = pos1 - pos2
        delta = np.where(delta > self.box_size / 2, delta - self.box_size, delta)
        delta = np.where(delta < -self.box_size / 2, delta + self.box_size, delta)
        return delta

    def compute_forces(self):
        # Reset forces to zero
        self.F.fill(0)
        # Compute pairwise forces
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                # Calculate distance vector
                r_ij = self.X[i] - self.X[j]
                # Apply periodic boundary conditions
                r_ij -= np.round(r_ij / self.box_size) * self.box_size
                r2 = np.dot(r_ij, r_ij)
                if r2 < 1.0:
                    # Calculate force based on distance
                    f = (1 - np.sqrt(r2)) * r_ij / r2
                    self.F[i] += f
                    self.F[j] -= f
        pass

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
            if step % 100 == 0:
                print(f"Step {step}/{num_steps} completed.")


def main():
    density = 4
    box_size = 15.0
    num_particles = density * box_size ** 2
    time_step = 0.01
    num_steps = 1000

    dpd_simulation = DPD(num_particles, box_size, time_step)
    start_time = time.time()
    dpd_simulation.run_simulation(num_steps)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
