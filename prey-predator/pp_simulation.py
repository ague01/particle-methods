import os
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


class Environment:
    def __init__(self, env_size=10, cell_size_lb=0.5):
        """Construct a cell-list based environment."""
        # For simplicity: cell-list have smallest cell size >= lower bound s.t. env_size % cell_size == 0
        self.env_size = env_size
        self.n_cells = int(env_size / cell_size_lb)
        self.cell_size = env_size / self.n_cells
        # Initialize the cell list, column major order
        self.cell_list = np.frompyfunc(
            list, 0, 1)(
            np.empty((self.n_cells * self.n_cells,),
                     dtype=object))  # type: ignore

    def add_agent(self, agent):
        """Add an agent to the cell list."""
        idx = self.get_cell_idx(agent.position)
        self.cell_list[idx].append(agent)

    def get_cell_idx(self, position):
        """Return the cell corresponding to a given real-valued position. Internal col major order.
        Account for periodic boundary conditions.
        """
        idx = np.floor((position % self.env_size) / self.cell_size)
        return int(idx[0] * self.n_cells + idx[1])

    def compact_lists(self):
        """Remove empty cells from the given list."""
        for index in range(len(self.cell_list)):
            self.cell_list[index] = [agent for agent in self.cell_list[index] if agent is not None]

    def get_neighbour_cells(self, position):
        """Return the indices of the neighbouring cells of a given position."""
        # Get neighbouring positions at distance cell_size along x axis
        pos_x = [(position - (self.cell_size, 0)) % self.env_size,
                 position,
                 (position + (self.cell_size, 0)) % self.env_size]
        pos = []
        # For each position found above, get the neighbouring positions at distance cell_size along y axis
        for p in pos_x:
            pos.append(p)
            pos.append((p + (0, self.cell_size)) % self.env_size)
            pos.append((p - (0, self.cell_size)) % self.env_size)
        # Get the corresponding cell indices
        return [self.get_cell_idx(p) for p in pos]

    def visualize(self):
        """Visualize the environment and the agents."""
        # Create a figure
        fig, ax = plt.subplots()
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')
        # Plot the grid
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                cell_idx = i * self.n_cells + j
                rect = plt.Rectangle((j * self.cell_size, i * self.cell_size),  # type: ignore
                                     self.cell_size, self.cell_size, fill=False)
                ax.add_patch(rect)
                ax.text(j * self.cell_size + 0.03, i * self.cell_size + 0.06, str(cell_idx),
                        fontsize=8, verticalalignment='top', horizontalalignment='left')
        # Plot the agents
        for cell in self.cell_list:
            for agent in cell:
                ax.plot(agent.position[0], agent.position[1], 'o', color='pink')
        # Display the plot
        plt.savefig('./out/env.png')


class Agent:
    _max_age = -1
    _sigma = -1
    _env_size = -1
    _reproduction_rate = -1

    def __init__(self, position, age=0, env_size=10):
        self.position = np.array(position)
        self.age = age
        if self._env_size == -1:
            self._env_size = env_size
        # Flag that is equal to time_step%2 when the agent has been updated
        # Agent is first evaluated and then evolved in time so initially flag=1
        self.updated: np.int8 = np.int8(1)

    def move(self):
        """Move the agent in a random direction by a given step size, accounting for periodic boundary conditions."""
        # Generate a random direction (half a sphere) and compute the corresponding displacement
        angle = rng.uniform(0, np.pi)
        # Generate a random step size from a normal distribution (can be negative)
        step_size = rng.normal(0, self._sigma)
        # Compute the displacement
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        # Update the position of the agent, accounting for periodic boundary conditions
        self.position += np.array([dx, dy])
        self.position %= self._env_size

    def make_age(self):
        """Perform a single time step of the agent."""
        self.age += 1

    def is_dead(self):
        return self.age >= self._max_age

    def distance_to(self, other):
        """Compute the squared Euclidean distance between two agents, accounting for periodic boundary conditions."""
        delta = np.abs(self.position - other.position)
        delta = np.where(delta > self._env_size / 2, self._env_size - delta, delta)
        return np.sum(np.square(delta))

    def can_update(self, time_step):
        """Return True if the agent can be updated and change flag, False otherwise."""
        if self.updated == time_step % 2:
            return False
        self.updated = time_step % 2
        return True


class Rabbit(Agent):
    _max_age = -1
    _sigma = -1

    def __init__(self, position, age=0, sigma=0.5, env_size=10, max_age=100, repr_rate=0.02):
        super().__init__(position, age, env_size)
        if self._max_age == -1:
            self._max_age = max_age
        if self._sigma == -1:
            self._sigma = 0.5
        if self._reproduction_rate == -1:
            self._reproduction_rate = repr_rate

    def replicate(self):
        if rng.random() < self._reproduction_rate:
            return Rabbit(self.position, age=0, sigma=self._sigma, max_age=self._max_age,
                          repr_rate=self._reproduction_rate, env_size=self._env_size)
        return None


class Wolf(Agent):
    _max_age = -1
    _sigma = -1
    _eat_rate = -1
    _sq_eat_radius = -1

    def __init__(
            self, position, age=0, sigma=0.5, env_size=10, max_age=50, repr_rate=0.02,
            eat_rate=0.02, eat_radius=0.5):
        """Initialize a Wolf object, subclass of Agent. Here age is the number of time steps since the wolf has eaten.

        Args:
            position (tuple): the initial position of the wolf.
            age (int, optional): the initial age. Defaults to 0.
            sigma (float, optional): the standard deviation of the normal distribution used to generate the step size. Defaults to 0.5.
            env_size (int, optional): environment size assuming a squared environment. Defaults to 10.
            max_age (int, optional): maximum age for the given class of agents. Defaults to 50.
        """
        super().__init__(position, age, env_size)
        if self._max_age == -1:
            self._max_age = max_age
        if self._sigma == -1:
            self._sigma = sigma
        if self._reproduction_rate == -1:
            self._reproduction_rate = repr_rate
        if self._eat_rate == -1:
            self._eat_rate = eat_rate
        if self._sq_eat_radius == -1:
            self._sq_eat_radius = eat_radius * eat_radius

    def replicate(self):
        if rng.random() < self._reproduction_rate:
            return Wolf(self.position, age=0, sigma=self._sigma, max_age=self._max_age,
                        repr_rate=self._reproduction_rate, eat_rate=self._eat_rate,
                        env_size=self._env_size)
        return None

    def eat(self, env):
        """Eat rabbits and return the number of eaten rabbits."""
        # Get the neighbouring cells
        neighbour_cells = env.get_neighbour_cells(self.position)
        # Initialize the number of eaten rabbits
        n_eaten_rabbits = 0
        n_available_rabbits = 0
        # For each neighbouring cell
        for cell_idx in neighbour_cells:
            # For each rabbit in the cell
            for r in range(len(env.cell_list[cell_idx])):
                rabbit = env.cell_list[cell_idx][r]
                if rabbit is None:
                    continue
                # If the rabbit is close enough to the wolf
                if self.distance_to(rabbit) < self._sq_eat_radius:
                    n_available_rabbits += 1
                    if rng.uniform() < self._eat_rate:
                        # Remove the rabbit from the cell list
                        env.cell_list[cell_idx][r] = None
                        # Increment the number of eaten rabbits
                        n_eaten_rabbits += 1
        # If the wolf has eaten, reset its age
        if n_eaten_rabbits > 0:
            self.age = 0
        return n_eaten_rabbits


def initialize(
        env_size, n_dims, n_rabbits, n_wolves,
        rabbit_max_age, rabbit_sigma, rabbit_replication_rate,
        wolf_max_age, wolf_sigma, wolf_replication_rate, wolf_eating_rate, wolf_eating_radius):
    # Initialize the environment
    env = Environment(env_size=env_size, cell_size_lb=wolf_eating_radius)
    print(
        f'Initialized environment with: size={env_size}, cell_size={env.cell_size}, n_cells={env.n_cells}')

    # Initialize agents
    # Generate list of random positions for rabbits and wolves
    positions = rng.random(size=(n_rabbits + n_wolves, n_dims)) * env_size
    # Generate list of ages for rabbits (wolves starts from age 0)
    ages = rng.integers(0, rabbit_max_age, size=n_rabbits)
    # Generate rabbit in environment
    for p, a in zip(positions[:n_rabbits], ages):
        env.add_agent(Rabbit(p, a, sigma=rabbit_sigma, env_size=env_size,
                             max_age=rabbit_max_age, repr_rate=rabbit_replication_rate))
    # Generate list of wolves
    wolves = [
        Wolf(
            p, 0, sigma=wolf_sigma, env_size=env_size, max_age=wolf_max_age,
            repr_rate=wolf_replication_rate, eat_rate=wolf_eating_rate,
            eat_radius=wolf_eating_radius)
        for p in positions[n_rabbits:]]

    return env, wolves


def start_evolution(env, wolves, max_time, n_init_rabbits, n_init_wolves):
    # Store number of rabbits and wolves at each time step
    tot_rabbits = [n_init_rabbits,]
    tot_wolves = [n_init_wolves,]

    for t in range(max_time):
        tot_rabbits.append(tot_rabbits[-1])
        tot_wolves.append(tot_wolves[-1])

        # Update rabbits
        for c in range(len(env.cell_list)):
            cell = env.cell_list[c]
            for i in range(len(cell)):
                rabbit = cell[i]
                # Check if already updated
                if not rabbit.can_update(t):
                    continue
                # Check if the rabbit is dead
                if rabbit.is_dead():
                    cell[i] = None
                    tot_rabbits[-1] -= 1
                    continue

                # Move the rabbit
                rabbit.move()
                # If rabbit is in a new cell, remove it from the previous cell
                if env.get_cell_idx(rabbit.position) != c:
                    cell[i] = None
                    env.add_agent(rabbit)

                # Check if the rabbit should replicate
                new_rabbit = rabbit.replicate()
                if new_rabbit is not None:
                    tot_rabbits[-1] += 1
                    env.add_agent(new_rabbit)
                    # Skip the update in the current time step
                    new_rabbit.can_update(t)

                # Age the rabbit
                rabbit.make_age()

        # Update wolves (reverse order since we may remove or append wolves)
        for wolf in wolves[:-1]:
            # Check if the wolf is dead
            if wolf.is_dead():
                wolves.remove(wolf)
                tot_wolves[-1] -= 1
                continue

            # Move the wolf
            wolf.move()

            # Eat rabbits
            n_eaten_rabbits = wolf.eat(env)
            tot_rabbits[-1] -= n_eaten_rabbits

            # Replicate the wolf for each eaten rabbit
            for _ in range(n_eaten_rabbits):
                new_wolf = wolf.replicate()
                if new_wolf is not None:
                    tot_wolves[-1] += 1
                    wolves.append(new_wolf)

            # Age the wolf
            wolf.make_age()

        # Compact the cell list for rabbits
        env.compact_lists()

        if tot_rabbits[-1] == 0 or tot_wolves[-1] == 0:
            break

        if t % 50 == 0:
            print(f'Time step: {t}, n_rabbits: {tot_rabbits[-1]}, n_wolves: {tot_wolves[-1]}')

    return tot_rabbits, tot_wolves


def main():
    env_size = 10
    n_dims = 2

    n_rabbits = 900
    n_wolves = 100

    rabbit_max_age = 100
    rabbit_sigma = 0.5
    rabbit_replication_rate = 0.02

    wolf_max_age = 50
    wolf_sigma = 0.5
    wolf_replication_rate = 0.02
    wolf_eating_rate = 0.02
    wolf_eating_radius = 0.5

    if not os.path.exists(f'./out'):
            os.makedirs(f'./out')

    for p in ('a', 'b', 'c'):

        if p == 'b':
            rabbit_max_age = 50
        if p == 'c':
            rabbit_max_age = 100
            rabbit_sigma = 0.05
            wolf_sigma = 0.05

        env, wolves = initialize(
            env_size, n_dims, n_rabbits, n_wolves,
            rabbit_max_age, rabbit_sigma, rabbit_replication_rate,
            wolf_max_age, wolf_sigma, wolf_replication_rate, wolf_eating_rate, wolf_eating_radius)

        rabbits, wolves = start_evolution(env, wolves, max_time=3500, n_init_rabbits=n_rabbits, n_init_wolves=n_wolves)

        # Plot the number of rabbits and wolves at each time step
        plt.figure()
        plt.plot(rabbits, label='Rabbits')
        plt.plot(wolves, label='Wolves')
        plt.legend()
        plt.grid()
        plt.xlabel('Time step')
        plt.ylabel('Number of agents')
        plt.title(f'Population evolution ({p})')
        plt.savefig(f'./out/population_{p}.png')

        # Plot the phase space with number of rabbits and wolves
        plt.figure()
        plt.plot(rabbits, wolves)
        plt.grid()
        plt.xlabel('Number of rabbits')
        plt.ylabel('Number of wolves')
        plt.title(f'Phase space ({p})')
        plt.savefig(f'./out/phase_space_{p}.png')

    return


if __name__ == '__main__':
    main()
