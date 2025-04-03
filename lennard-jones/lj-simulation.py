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

    _r_cutoff: float = -1.
    _mass: float = -1.

    def __init__(self, position: np.ndarray, velocity: np.ndarray, r_cutoff: float, mass: float, dt: float):
        """Initialize a particle."""
        if Particle._r_cutoff < 0:
            Particle._r_cutoff = r_cutoff
        if Particle._mass < 0:
            Particle._mass = mass

        self.position: np.ndarray = position
        self.velocity: np.ndarray = velocity
        self.previous_position: np.ndarray = position - velocity * dt
        self.force = np.zeros_like(position)

    def update_position(self, dt: float, environment: 'Environment'):
        """Update the position of the particle using the current velocity."""
        self.position += self.velocity * dt
        # Apply periodic boundary conditions
        self.position %= environment.env_size

    def update_force(self, environment: 'Environment'):
        """Update the force on the particle."""
        self.force = self.force * 1


class Environment:
    def __init__(self, env_size: float = 10, cell_size_lb: float = 0.5):
        """Construct a cell-list based domain with periodic boundary conditions."""
        # For simplicity: cell-list have smallest cell size >= lower bound s.t. env_size % cell_size == 0
        self.env_size = env_size
        self.n_cells = int(env_size / cell_size_lb)
        self.cell_size = env_size / self.n_cells

        # Initialize the cell linked list
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

    def periodic_distance(self, pos1, pos2):
        """Compute the periodic distance between two positions."""
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > self.env_size / 2, self.env_size - delta, delta)
        return np.sum(np.square(delta)) ** 0.5

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


def initialize(n_particles: int, domain_size: float, n_dims: int, T: float, mass: float, r_cutoff: float, dt: float) -> Environment:
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
        particle = Particle(positions[i], velocities[i], r_cutoff, mass, dt)
        env.add_particle(particle)

    # Plot initial positions and velocities
    fig, ax = plt.subplots()
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        angles='xy', scale_units='xy', scale=1)
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    ax.set_title('Initial Positions and Velocities')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.savefig('./out/init_positions_velocities.png')
    plt.close(fig)

    # Plot the distribution of velocities
    fig, ax = plt.subplots()
    ax.hist(np.linalg.norm(velocities, axis=1), bins=30, density=True)
    ax.set_title('Velocity Distribution')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Density')
    plt.savefig('./out/init_velocity_dist.png')
    plt.close(fig)

    return env

def simulate(env: Environment, dt: float, max_time: int):
    t = 0
    while t < max_time:
        # Update particle positions
        for cell in env.cells.values():
            for particle in cell:
                particle.update_position(dt, env)

        # Update forces
        for cell in env.cells.values():
            for particle in cell:
                particle.update_force(env)

        # Update velocities based on forces
        for cell in env.cells.values():
            for particle in cell:
                particle.velocity += particle.force * dt / Particle._mass

        # Increment time
        t += dt


def main():
    n_dims = 2
    domain_size = 10.

    n_particles = 500

    mass = 1.
    T = 0.5
    r_cutoff = 2.5

    max_time = 100
    dt = 0.01

    # Initialize the environment and particles
    env = initialize(n_particles, domain_size, n_dims, T, mass, r_cutoff, dt)
    print('Simulation initialized.')
    print('Total momentum:', env.compute_total_momentum())
    #simulate(env, dt, max_time)



if __name__ == '__main__':
    main()
