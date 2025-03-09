import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Initialize the Marsenne Twister random number generator
seed = 65647437888080803208636358831803839626
rng = np.random.default_rng(np.random.MT19937(seed))

# Set matplotlib backend for non interactive plotting
mpl.use('Agg')

# Avoid warning when dividing by zero (T=0)
np.seterr(divide='ignore', invalid='ignore')


def get_neighbours(a, b, size, avoid_repeat=False):
    # Avoid counting each pair twice
    if avoid_repeat:
        return [[a, (b + 1) % size[1]],
                [(a + 1) % size[0], b]]
    else:
        return [[(a + 1) % size[0], b],
                [a, (b + 1) % size[1]],
                [(a - 1) % size[0], b],
                [a, (b - 1) % size[1]]]


def random_initial_state(size):
    return 2 * rng.integers(2, size=size) - 1


def compute_energy(state, coupling):
    energy = - coupling
    energy *= sum(state[a, b] * state[x, y]
                  for a in range(state.shape[0])
                  for b in range(state.shape[1])
                  for x, y in get_neighbours(a, b, state.shape, avoid_repeat=True))
    return energy


def compute_energy_diff(state, a, b, coupling):
    h = sum(state[x, y] for x, y in get_neighbours(a, b, state.shape))
    return 2 * coupling * state[a, b] * h


def metropolis(state, temperature, n_steps, coupling, kB):
    for _ in range(n_steps):
        # Pick a random site in the lattice
        a, b = (rng.random(size=2) * state.shape).astype(int)
        # Compute the energy difference
        delta_E = compute_energy_diff(state, a, b, coupling)
        # Metropolis acceptance criterion
        if (delta_E < 0) or (rng.random() < np.exp(-delta_E / kB / temperature)):
            state[a, b] *= -1
    return state


def simulate(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps):
    # Initialize the lattice with random spins
    state = random_initial_state(size)
    # Run the metropolis algorithm for n_therm steps
    state = metropolis(state, temperature, n_steps=n_therm, coupling=coupling, kB=kB)

    magnetizations = []
    energies = []
    # Run the metropolis algorithm for n_samples samples
    for _ in range(n_samples):
        # Run the metropolis algorithm for n_subsweeps steps
        state = metropolis(state, temperature, n_steps=n_subsweeps, coupling=coupling, kB=kB)
        # Measure the magnetization
        magnetizations.append(np.mean(state))
        # Measure the energy
        energies.append(compute_energy(state, coupling))

    # Compute quantities
    avg_mag = np.mean(np.abs(magnetizations))
    std_dev_mag = np.std(magnetizations)
    avg_energy = np.mean(energies)
    std_dev_energy = np.std(energies)

    return avg_mag, std_dev_mag, avg_energy, std_dev_energy


def main():
    out_path = './out/'
    os.makedirs(out_path, exist_ok=True)
    sizes = [(L, L) for L in [5, 10, 15]]
    temperatures = [0.0, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]
    n_therm = 100_000
    n_samples = 5_000
    coupling = 1.0
    kB = 1.0

    print('Markov Chain Ising Model Simulation')
    print('-'*30)
    print(f'Sizes: {sizes}')
    print(f'Coupling: {coupling}')
    print(f'Temperatures: {temperatures}')
    print(f'Thermalization Steps: {n_therm}')
    print(f'Number of Samples: {n_samples}')
    print('-'*30)

    # a) determine the critical temperature
    # b) plot the magnetization as a function of temperature for different sizes
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 10))
    for size in sizes:
        n_subsweeps = size[0] * size[1]
        print(f'Size: {size} (subsweeps={n_subsweeps})')

        magnetizations = []
        energies = []
        cvs = []
        chis = []

        for temperature in temperatures:
            m, sdm, e, sde = simulate(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps)
            magnetizations.append(m)
            energies.append(e)
            chis.append(sdm / (temperature * size[0] * size[1]))
            cvs.append(sde / (temperature**2 * size[0] * size[1]))
            print(f'\tTemperature {temperature}: Magnetization={m:.4f}({sdm:.4f}), Energy={e:.4f}({sde:.4f})')

        # Print the critical temperature
        print(f'Critical Temperature for L={size[0]}: {temperatures[np.argmax(cvs)]}')

        # Plot the magnetization as a function of temperature for size = L
        ax[0, 0].plot(temperatures, magnetizations, 'o-', label=f'L={size[0]}')
        ax[0, 1].plot(temperatures, energies, 'o-', label=f'L={size[0]}')
        ax[1, 0].plot(temperatures, cvs, 'o-', label=f'L={size[0]}')
        ax[1, 1].plot(temperatures, chis, 'o-', label=f'L={size[0]}')

    ax[0, 0].set_ylabel('Average Magnetization')
    ax[0, 0].set_title('Magnetization vs Temperature')
    ax[0, 0].grid()

    ax[0, 1].set_xlabel('Temperature')
    ax[0, 1].set_ylabel('Average Energy')
    ax[0, 1].set_title('Energy vs Temperature')
    ax[0, 1].grid()

    ax[1, 0].set_xlabel('Temperature')
    ax[1, 0].set_ylabel('Heat Capacity $C_v$')
    ax[1, 0].set_title('Heat Capacity vs Temperature')
    ax[1, 0].grid()

    ax[1, 1].set_xlabel('Temperature')
    ax[1, 1].set_ylabel('Susceptibility $\chi$')
    ax[1, 1].set_title('Susceptibility vs Temperature')
    ax[1, 1].grid()

    # Plot vertical lines at critical temperature
    ax[0, 0].axvline(x=2.269, color='r', linestyle='--', label='$T_C$')
    ax[0, 1].axvline(x=2.269, color='r', linestyle='--')
    ax[1, 0].axvline(x=2.269, color='r', linestyle='--')
    ax[1, 1].axvline(x=2.269, color='r', linestyle='--')

    fig.legend()
    plt.tight_layout()
    plt.savefig(out_path + f'magnetization_energy.png')


if __name__ == '__main__':
    main()
