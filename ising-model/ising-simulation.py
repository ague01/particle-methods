import numpy as np
from numpy.random import MT19937
from numpy.random import Generator
import matplotlib.pyplot as plt

# Initialize the Marsenne Twister random number generator
seed = 65647437836358831880808032086803839626
rng = np.random.default_rng(MT19937(seed))

# Avoid warning when dividing by zero (T=0)
np.seterr(divide='ignore')


def get_neighbours(a, b, size):
    return [[(a + 1) % size[0], b],
            [a, (b + 1) % size[1]],
            [(a - 1) % size[0], b],
            [a, (b - 1) % size[1]]]


def random_initial_state(size):
    return 2 * np.random.randint(2, size=size) - 1


def compute_energy(state, coupling):
    energy = - coupling
    energy *= sum(state[a, b] * state[x, y]
                  for a in range(state.shape[0])
                  for b in range(state.shape[1])
                  for x, y in get_neighbours(a, b, state.shape))
    return energy


def compute_energy_diff(state, a, b, coupling):
    h = sum(state[x, y] for x, y in get_neighbours(a, b, state.shape))
    return 2 * coupling * state[a, b] * h


def average_magnetization(state):
    return np.mean(state)


def time_average(quantities):
    return np.mean(quantities)


def metropolis(state, temperature, n_steps, coupling, kB):
    for _ in range(n_steps):
        # Pick a random site in the lattice
        a, b = (rng.random(size=2) * state.shape).astype(int)
        # Compute the energy difference
        delta_E = compute_energy_diff(state, a, b, coupling)
        # Metropolis acceptance criterion
        if (delta_E < 0) or (np.random.random() < np.exp(-delta_E / kB / temperature)):
            state[a, b] *= -1
    return state


def simulate(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps):
    # Initialize the lattice with random spins
    state = random_initial_state(size)
    # Run the metropolis algorithm for n_them steps
    state = metropolis(state, temperature, n_steps=n_therm, coupling=coupling, kB=kB)
    # Initialize the quantities to be measured
    magnetizations = []
    energy = []
    # Run the metropolis algorithm for n_samples samples
    for _ in range(n_samples):
        # Run the metropolis algorithm for n_subsweeps steps
        state = metropolis(state, temperature, n_steps=n_subsweeps, coupling=coupling, kB=kB)
        # Measure the magnetization
        magnetizations.append(np.abs(average_magnetization(state)))
        # Measure the energy
        energy.append(compute_energy(state, coupling))
    # Compute the time average of the magnetization
    magnetization = time_average(magnetizations)
    energy = time_average(energy)

    return magnetization, energy


def main():
    out_path = './out/'

    size = (10, 10)
    temperatures = [0.0, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]
    n_therm = 100_000
    n_samples = 5_000
    n_subsweeps = size[0] * size[1]
    coupling = 1.0
    kB = 1.0

    print('Marcov Chain Ising Model Simulation')
    print('-'*30)
    print(f'Size: {size}')
    print(f'Coupling: {coupling}')
    print(f'Temperatures: {temperatures}')
    print(f'Thermalization Steps: {n_therm}')
    print(f'Number of Samples: {n_samples}')
    print(f'Subsweeps per Sample: {n_subsweeps}')
    print('-'*30)

    magnetizations = []
    energies = []
    for temperature in temperatures:
        print(f'Temperature: {temperature}')
        m, e = simulate(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps)
        magnetizations.append(m)
        energies.append(e)
        print(f'\tMagnetization: {m}')
        print(f'\tEnergy: {e}')


if __name__ == '__main__':
    main()
