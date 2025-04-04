import os
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Initialize the Marsenne Twister random number generator
seed = 65647437888080803208636358821843839626
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
        acceleration = self.force / Particle._mass
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        self.position %= environment.env_size

    def update_force_velocity(self, dt: float, environment: 'Environment'):
        """Update the particle force and the velocity based on the velocity Verlet algorithm."""
        old_force : np.ndarray = self.force.copy()
        # Update force using the current position (already updated)
        self.update_force(environment)
        # Update velocity
        self.velocity += 0.5 * (old_force + self.force) / Particle._mass * dt

    def update_force(self, environment: 'Environment'):
        """Update the force on the particle."""
        self.force = self.compute(environment, quantity='force') # type: ignore

    def compute(self, environment: 'Environment', quantity: str, n_particles: int = -1) -> float | np.ndarray:
        """Compute a quantity for the particle. ('force' or 'potential')"""
        if quantity == 'force':
            # Compute the force on the particle by another particle without the 48 factor
            def func(pos, r2, r6): return pos / r2 / r6 * (1/r6 - 0.5)
        elif quantity == 'potential':
            if n_particles < 0:
                raise ValueError("Number of particles must be provided for potential energy calculation.")
            # Compute the potential energy of the particle by another particle
            def func(pos, r2, r6): return 1 / r6 * (1/r6 - 1)
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

        ret = 0.0
        # Compute quantity from other particles in the same cell and neighbouring cells
        for neighbour_cell in environment.get_neighbour_cells(self.position):
            for particle in environment.cells[neighbour_cell]:
                r2 = environment.periodic_squared_distance(self.position, particle.position)
                if r2 < Particle._sq_r_cutoff and r2 > 0:
                    # Compute the squared distance between particles
                    ret += func(self.position, r2, r2 ** 3)

        # Add factors
        if quantity == 'force':
            ret *= 48
        elif quantity == 'potential':
            ecut = func(self.position, Particle._sq_r_cutoff, Particle._sq_r_cutoff ** 3)
            ret -= ecut * n_particles

        return ret

class Environment:
    def __init__(self, env_size: float = 10, cell_size_lb: float = 0.5):
        """Construct a cell-list based domain with periodic boundary conditions."""
        # For simplicity: cell-list have smallest cell size >= lower bound s.t. env_size % cell_size == 0
        self.env_size = env_size
        self.n_cells = int(env_size / cell_size_lb)
        self.cell_size = env_size / self.n_cells

        # Initialize the cell-list structure
        self.cells = {(i, j): [] for i in range(self.n_cells) for j in range(self.n_cells)}

    def add_particle(self, particle: Particle):
        """Add a particle to the cell list."""
        idx = self.get_cell_idx(particle.position)
        self.cells[idx].append(particle)

    def get_cell_idx(self, position: np.ndarray):
        """Return the cell tuple index for a given real-valued position accounting for periodic boundaries."""
        return tuple((position // self.cell_size).astype(int) % self.n_cells)

    def get_neighbour_cells(self, position):
        """Return the indices of the neighbouring cells of a given position."""
        idx = self.get_cell_idx(position)
        return [
            ((idx[0] + dx) % self.n_cells, (idx[1] + dy) % self.n_cells)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
        ]

    def periodic_squared_distance(self, pos1, pos2):
        """Compute the periodic squared distance between two positions."""
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > self.env_size / 2, self.env_size - delta, delta)
        return np.sum(np.square(delta))

    def periodic_position(self, pos):
        """Compute the true position in the periodic domain given a position."""
        return pos % self.env_size

    def compute_total_momentum(self) -> np.ndarray:
        """Compute the total momentum of the system."""
        total_momentum = np.zeros(2)
        for cell in self.cells.values():
            for particle in cell:
                total_momentum += particle._mass * particle.velocity
        return total_momentum

    def compute_total_energies(self) -> tuple[float, float]:
        """Compute the total kinetic and potential energy of the system."""
        t_kinetic_energy = 0.0
        t_potential_energy = 0.0
        for cell in self.cells.values():
            for particle in cell:
                # Kinetic energy
                t_kinetic_energy += np.sum(np.square(particle.velocity))
                # Potential energy (from acceleration)
                t_potential_energy += np.sum(np.square(particle.force)) / 2

        t_kinetic_energy *= 0.5 * Particle._mass

        return t_kinetic_energy, t_potential_energy

    def plot_particles(self, file_id: str = ''):
        """Plot the particles in the environment."""
        # Extract positions and velocities from particles
        positions = []
        velocities = []
        for cell in self.cells.values():
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


def initialize(
        n_particles: int, domain_size: float, n_dims: int, T: float, mass: float, r_cutoff: float,
        dt: float) -> Environment:
    """Initialize the simulation with random particles."""
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
        particle = Particle(positions[i], velocities[i], r_cutoff, mass)
        env.add_particle(particle)

    # Update the particle forces based on their initial positions
    for cell in env.cells.values():
        for particle in cell:
            particle.update_force(env)

    return env


def simulate(env: Environment, dt: float, max_time: float):
    t = 0
    while t < max_time:
        if t % 0.1 < dt:
            print(f'Starting simulation step n={t/dt:.0f} at time t={t:.2f}')
        # Update particle positions and velocities
        for cell in env.cells.values():
            for particle in cell:
                particle.update_position(dt, env)
        # Update cell list
        for idx, cell in env.cells.items():
            for particle in cell:
                new_cell_idx = env.get_cell_idx(particle.position)
                if new_cell_idx != idx:
                    # Remove from old cell
                    env.cells[idx].remove(particle)
                    # Add to new cell
                    env.cells[new_cell_idx].append(particle)

        # Update forces
        for cell in env.cells.values():
            for particle in cell:
                particle.update_force(env)

        # Increment time
        t += dt


def main():
    n_dims = 2
    domain_size = 10.

    n_particles = 200
    mass = 1.
    T = 0.5
    r_cutoff = 2.5

    max_time = 1.
    dt = 0.01

    # Initialize the environment and particles
    env = initialize(n_particles, domain_size, n_dims, T, mass, r_cutoff, dt)
    env.plot_particles(file_id='initial')
    print('Simulation initialized.')
    print('Initial total momentum:', env.compute_total_momentum())

    # Simulate the environment
    simulate(env, dt, max_time)
    env.plot_particles(file_id='final')
    print('Simulation completed.')
    print('Final total momentum:', env.compute_total_momentum())


if __name__ == '__main__':
    main()
