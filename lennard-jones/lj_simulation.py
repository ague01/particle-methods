from math import e
import os
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Initialize the Marsenne Twister random number generator
seed = 65647437888080846208636358821643839626
rng = np.random.default_rng(np.random.MT19937(seed))

# Set matplotlib backend for non interactive plotting
mpl.use('Agg')

# Avoid warning when dividing by zero (T=0)
# np.seterr(divide='ignore', invalid='ignore')


class Particle:
    """Class representing a particle in the simulation."""

    _sq_r_cutoff: float = -1.
    _mass: float = -1.

    def __init__(self, position: np.ndarray, velocity: np.ndarray, r_cutoff: float, mass: float):
        """Initialize a particle."""
        if Particle._sq_r_cutoff < 0:
            Particle._sq_r_cutoff = r_cutoff ** 2
        if Particle._mass < 0:
            Particle._mass = mass

        self.position: np.ndarray = position
        self.velocity: np.ndarray = velocity
        self.force: np.ndarray = np.zeros_like(position)

    def update_position(self, dt: float, environment: 'Environment'):
        """Update the particle position based on the velocity Verlet algorithm."""
        self.position += self.velocity * dt + 0.5 * (self.force / Particle._mass) * (dt ** 2)
        self.position = environment.periodic_position(self.position)

    def update_velocity(self, dt: float):
        """Partially update the particle velocity based on the force before/after its update."""
        self.velocity += 0.5 * (self.force / Particle._mass) * dt

    def reset_force(self):
        """Update the particle force and the velocity based on the velocity Verlet algorithm."""
        self.force = np.zeros_like(self.force)


class Environment:
    def __init__(self, env_size: float = 10, cell_size_lb: float = 0.5):
        """Construct a cell-list based domain with periodic boundary conditions."""
        # For simplicity: cell-list have smallest cell size >= lower bound s.t. env_size % cell_size == 0
        self.env_size = env_size  # Size of the environment in each dimension
        self.n_cells = int(env_size / cell_size_lb)  # Number of cells in each dimension
        self.cell_size = env_size / self.n_cells  # Size of each cell in each dimension

        # Initialize the cell-list structure (cell (i,j) accessed as i*n_cells + j)
        self.cells: list[list] = [[] for _ in range(self.n_cells * self.n_cells)]

    def add_particle(self, particle: Particle):
        """Add a particle to the correct cell list based on its position."""
        idx = self.get_cell_idx(particle.position)
        self.cells[idx].append(particle)

    def get_cell_idx(self, position: np.ndarray) -> int:
        """Return the flat cell index for a given real-valued position accounting for periodic boundaries."""
        i, j = tuple((position // self.cell_size).astype(int) % self.n_cells)
        return i * self.n_cells + j

    def get_neighbour_cells(self, cell_idx: int) -> list[int]:
        """Return the list of flatten indices of the neighbouring cells of a given cell avoiding double counting."""
        idx = (cell_idx // self.n_cells, cell_idx % self.n_cells)
        neighbours = [
            ((idx[0] + dx) % self.n_cells, (idx[1] + dy) % self.n_cells)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
        ]
        # Flatten the list of tuples and remove duplicates
        neighbours = list(set(i * self.n_cells + j for i, j in neighbours))
        # Remove the index less than the current cell
        neighbours = [n for n in neighbours if n > cell_idx]
        # Sort the list of indices
        neighbours.sort()
        return neighbours

    def periodic_squared_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute the periodic squared distance between two positions."""
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > self.env_size / 2, self.env_size - delta, delta)

        return np.sum(np.square(delta))

    def periodic_position(self, pos) -> np.ndarray:
        """Compute the true position in the periodic domain given a position."""
        return pos % self.env_size

    def periodic_displacement(self, pos1, pos2) -> np.ndarray:
        """Compute the displacement vector between two positions accounting for periodic boundaries."""
        delta = pos1 - pos2
        delta = np.where(delta > self.env_size / 2, delta - self.env_size, delta)
        delta = np.where(delta < -self.env_size / 2, delta + self.env_size, delta)
        return delta

    def update_cell_list(self):
        """Update cell-list based on the new position of particles."""
        for idx_c, cell in enumerate(self.cells):
            for idx_p, particle in enumerate(cell):
                new_idx_c = self.get_cell_idx(particle.position)
                if new_idx_c != idx_c:
                    # Flag the particle for moving to a new cell
                    self.cells[new_idx_c].append(particle)
                    cell[idx_p] = None  # Mark the old cell entry as None
            # Remove None entries
            self.cells[idx_c] = [p for p in cell if p is not None]

    def compute_total_momentum(self) -> np.ndarray:
        """Compute the total momentum vector of the system."""
        total_momentum = np.zeros(2)
        for cell in self.cells:
            for particle in cell:
                total_momentum += particle._mass * particle.velocity
        return total_momentum

    def compute_update(self) -> tuple[float, float]:
        """Compute the total kinetic and potential energy of the system and return them. Update the particles' force.

        This method has to be called after the particles' positions have been partially updated and the forces have been zeroed.
        """
        t_kinetic_energy = 0.
        t_potential_energy = 0.

        for cell in self.cells:
            for particle in cell:
                assert np.all(particle.force == 0), "Particle force should be initialized to zero."

        def func_pot(r6): return 4. / r6 * (1./r6 - 1.)
        ecut = func_pot(Particle._sq_r_cutoff ** 3)

        def func_for(r2, r6): return 48. / r2 / r6 * (1./r6 - 0.5)

        for i_cell, cell1 in enumerate(self.cells):
            # Compute same particle quantities
            for i_particle, particle1 in enumerate(cell1):
                # a) Particle Kinetic energy
                t_kinetic_energy += 0.5 * Particle._mass * np.sum(np.square(particle1.velocity))

                # Compute same cell interactions
                for particle2 in cell1[i_particle+1:]:
                    r2 = self.periodic_squared_distance(particle1.position, particle2.position)
                    if r2 <= Particle._sq_r_cutoff and r2 > 1e-16:
                        # b) Potential energy
                        t_potential_energy += (func_pot(r2 ** 3) - ecut)
                        # c) Force
                        delta_r = self.periodic_displacement(particle1.position, particle2.position)
                        ff = func_for(r2, r2 ** 3) * delta_r
                        particle1.force += ff
                        particle2.force -= ff

                #Compute neighbouring cell interactions
                for idx_cell2 in self.get_neighbour_cells(i_cell):
                    for particle2 in self.cells[idx_cell2]:
                        assert particle2 is not particle1
                        r2 = self.periodic_squared_distance(particle1.position, particle2.position)
                        if r2 <= Particle._sq_r_cutoff and r2 > 1e-16:
                            # b) Potential energy
                            t_potential_energy += (func_pot(r2 ** 3) - ecut)
                            # c) Force
                            delta_r = self.periodic_displacement(
                                particle1.position, particle2.position)
                            ff = func_for(r2, r2 ** 3) * delta_r
                            particle1.force += ff
                            particle2.force -= ff

        return t_kinetic_energy, t_potential_energy

    def plot_particles(self, file_id: str = ''):
        """Plot the particles in the environment."""
        # Extract positions and velocities from particles
        positions = []
        velocities = []
        for cell in self.cells:
            for particle in cell:
                positions.append(particle.position)
                velocities.append(particle.velocity)
        positions = np.array(positions)
        velocities = np.array(velocities)

        # Plot initial positions and velocities
        fig, ax = plt.subplots()
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            velocities[:, 0],
            velocities[:, 1],
            angles='xy', scale_units='xy', scale=1)
        ax.set_xlim(0, self.env_size)
        ax.set_ylim(0, self.env_size)
        ax.set_aspect('equal')
        ax.set_title(f'Positions and Velocities ({file_id})')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        plt.savefig(f'./out/{file_id}_positions_velocities.png')
        plt.close(fig)

        # Plot the distribution of velocities
        fig, ax = plt.subplots()
        ax.hist(np.linalg.norm(velocities, axis=1), bins=30, density=True)
        ax.set_title(f'Velocity Distribution ({file_id})')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Density')
        plt.savefig(f'./out/{file_id}_velocity_dist.png')
        plt.close(fig)

    def plot_forces(self, file_id: str = ''):
        """Plot the force distribution of the particles."""
        # Extract forces and energies from particles
        forces = []
        for cell in self.cells:
            for particle in cell:
                forces.append(particle.force)
        forces = np.array(forces)

        # Plot the distribution of forces
        fig, ax = plt.subplots()
        ax.hist(np.linalg.norm(forces, axis=1), bins=30, density=True)
        ax.set_title(f'Force Distribution ({file_id})')
        ax.set_xlabel('Force')
        ax.set_ylabel('Density')
        plt.savefig(f'./out/{file_id}_force_dist.png')
        plt.close(fig)
        # Plot the distribution of momentum
        fig, ax = plt.subplots()


def initialize(
        n_particles: int, domain_size: float, n_dims: int, T: float, mass: float, r_cutoff: float,
        dt: float) -> Environment:
    """Initialize the simulation with random particles."""

    init_start_time = time.time()

    # Randomly initialize positions within the domain
    positions = rng.uniform(0, domain_size, size=(n_particles, n_dims))
    # Randomly initialize velocities according to the Maxwell-Boltzmann distribution at a given T
    velocities = rng.random(size=(n_particles, n_dims)) - 0.5
    # Compute the rescaling factor based on kinetic energy
    scale_f = np.sqrt(n_dims * T / np.mean(np.square(velocities), axis=0))
    # Remove center-of-mass velocity to ensure zero total momentum and correct scaling
    velocities = (velocities - np.mean(velocities, axis=0)) * scale_f

    # Create the environment and add particles
    env = Environment(env_size=domain_size, cell_size_lb=r_cutoff)
    for i in range(n_particles):
        env.add_particle(Particle(positions[i], velocities[i], r_cutoff, mass))

    # Update the particles' force based on their initial position
    env.compute_update()

    print(f"Initialization completed in {time.time() - init_start_time:.2f} seconds.")

    return env


def simulate(env: Environment, dt: float, max_time: float) -> tuple[list, list]:
    start_time = time.time()  # Start timing the simulation
    t = 0
    total_kinetic_energy = []
    total_potential_energy = []

    while t < max_time:
        if t % 0.1 < dt:
            print(f'Starting simulation step n={t/dt:.0f} at time t={t:.2f}')
        total_momentum = env.compute_total_momentum()
        print(
            f"Time {t:.2f}: Total Momentum = {total_momentum}, Magnitude = {np.linalg.norm(total_momentum)}")

        # Update particle positions and partially velocities
        for cell in env.cells:
            for particle in cell:
                particle.update_position(dt, env)
                particle.update_velocity(dt)

        # Update cell list based on new positions
        env.update_cell_list()

        # Reset forces
        for cell in env.cells:
            for particle in cell:
                particle.reset_force()

        # Compute new forces to update particle velocities and get total energies
        t_kin, t_pot = env.compute_update()
        total_kinetic_energy.append(t_kin)
        total_potential_energy.append(t_pot)

        # Finalize the particles' velocity update based on the new forces
        for cell in env.cells:
            for particle in cell:
                particle.update_velocity(dt)

        # Increment time
        t += dt

    # Print simulation duration
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")

    return total_kinetic_energy, total_potential_energy


def main():
    n_dims = 2
    domain_size = 2.5

    n_particles = 4
    mass = 1.
    T = 0.8
    r_cutoff = 2.5

    max_time = 0.1
    dt = 0.01

    # Initialize the environment and particles
    env = initialize(n_particles, domain_size, n_dims, T, mass, r_cutoff, dt)
    env.plot_particles(file_id='initial')
    env.plot_forces(file_id='initial')
    # Store the total momentum and energies
    initial_momentum = env.compute_total_momentum()

    min_distance = np.inf
    for cell in env.cells:
        for p1 in cell:
            for p2 in cell:
                if p1 is not p2:
                    r2 = env.periodic_squared_distance(p1.position, p2.position)
                    min_distance = min(min_distance, np.sqrt(r2))
    print(f"Minimum distance between particles: {min_distance:.2f}")

    # Simulate the environment
    t_kin, t_pot = simulate(env, dt, max_time)

    # Plot the energies as function of time
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(t_kin)) * dt, t_kin, label='Kinetic Energy')
    ax.plot(np.arange(len(t_pot)) * dt, t_pot, label='Potential Energy')
    ax.plot(np.arange(len(t_kin)) * dt, np.array(t_kin) + np.array(t_pot), label='Total Energy')
    ax.set_title('Kinetic and Potential Energy over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, which='both')
    plt.savefig('./out/energies.png')
    plt.close(fig)

    env.plot_particles(file_id='final')
    final_momentum = env.compute_total_momentum()

    # Print the difference in total momentum and energies
    print("Initial Total Momentum:", initial_momentum)
    print("Final Total Momentum:", final_momentum)
    print(f"Initial Total Energy: {t_kin[0] + t_pot[0]:e}")
    print(f"Final Total Energy: {t_kin[-1] + t_pot[-1]:e}")


if __name__ == '__main__':
    # Run the main simulation
    main()
