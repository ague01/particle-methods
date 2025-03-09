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
        a = rng.integers(state.shape[0])
        b = rng.integers(state.shape[1])
        # Compute the energy difference
        delta_E = compute_energy_diff(state, a, b, coupling)
        # Metropolis acceptance criterion
        if (delta_E < 0) or (rng.random() < np.exp(-delta_E / kB / temperature)):
            state[a, b] *= -1
    return state


def simulate_raw(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps):
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

    return magnetizations, energies


def simulate(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps):
    magnetizations, energies = simulate_raw(
        size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps)

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
    temperatures = [1e-6, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]
    n_therm = 100_000
    n_samples = 5_000
    coupling = 1.0
    kB = 1.0

    print('Markov Chain Ising Model Simulation for a) and b)')
    print('-'*30)
    print(f'Sizes: {sizes}')
    print(f'Coupling: {coupling}')
    print(f'Temperatures: {temperatures}')
    print(f'Thermalization Steps: {n_therm}')
    print(f'Number of Samples: {n_samples}')
    print('-'*30)
    '''
    # a) determine the critical temperature
    # b) plot the magnetization as a function of temperature for different sizes
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 10))
    for i, size in enumerate(sizes):
        n_subsweeps = size[0] * size[1]
        print(f'Size: {size} (subsweeps={n_subsweeps})')

        magnetizations = []
        energies = []
        cvs = []
        chis = []

        for temperature in temperatures:
            m, sdm, e, sde = simulate(size, coupling, kB, temperature,
                                      n_therm, n_samples, n_subsweeps)
            magnetizations.append(m)
            energies.append(e)
            chis.append(sdm / (temperature * size[0] * size[1]))
            cvs.append(sde / (temperature**2 * size[0] * size[1]))
            print(
                f'\tTemperature {temperature}: Magnetization={m:.4f}({sdm:.4f}), Energy={e:.4f}({sde:.4f})')

        # Print the critical temperature
        print(
            f'Critical Temperature for L={size[0]} based on capacity: {temperatures[np.argmax(cvs)]}')
        print(
            f'Critical Temperature for L={size[0]} based on susceptibility: {temperatures[np.argmax(chis)]}')

        # Plot quantities as a function of temperature for size = L
        col = ['tab:blue', 'tab:orange', 'tab:green']
        ax[0, 0].plot(temperatures, magnetizations, 'o-', color=col[i], label=f'L={size[0]}')
        ax[0, 1].plot(temperatures, energies, 'o-', color=col[i])
        ax[1, 0].plot(temperatures, cvs, 'o-', color=col[i])
        ax[1, 1].plot(temperatures, chis, 'o-', color=col[i])
        # Plot critical temperatures
        ax[1, 0].axvline(x=temperatures[np.argmax(cvs)], color=col[i], linestyle='dotted', label=f'$T_c$ L={size[0]}')
        ax[1, 1].axvline(x=temperatures[np.argmax(chis)], color=col[i], linestyle='dotted')

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
    ax[1, 1].set_ylabel(r'Susceptibility $\chi$')
    ax[1, 1].set_title('Susceptibility vs Temperature')
    ax[1, 1].grid()

    fig.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(out_path + f'magnetization_energy.png')
    '''
    # c) plot the dependence of M on the simulation time for T < Tc, small size
    size = (5, 5)
    n_subsweeps = size[0] * size[1]
    temperatures = [1.0, 1.5, 2.0, 2.1, 2.2]
    n_samples = 5_000
    print('\n')
    print('Markov Chain Ising Model Simulation for c)')
    print('-'*30)
    print(f'Size: {size}')
    print(f'Coupling: {coupling}')
    print(f'Temperatures: {temperatures}')
    print(f'Thermalization Steps: {n_therm}')
    print(f'Number of Samples: {n_samples}')
    print(f'Number of Subsweeps: {n_subsweeps}')
    print('-'*30)

    a = len(temperatures)//2 + 1
    _, ax = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
    for temperature in temperatures[:a]:
        magnetizations, _ = simulate_raw(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps)
        ax[0].plot(range(n_samples), magnetizations, '--', linewidth=1.0, label=f'T={temperature}')
    for temperature in temperatures[a:]:
        magnetizations, _ = simulate_raw(size, coupling, kB, temperature, n_therm, n_samples, n_subsweeps)
        ax[1].plot(range(n_samples), magnetizations, '--', linewidth=1.0, label=f'T={temperature}')
    ax[0].set_xlabel('Simulation Time')
    ax[1].set_xlabel('Simulation Time')
    ax[0].set_ylabel('Magnetization')
    ax[1].set_ylabel('Magnetization')
    ax[0].set_title('Magnetization vs Simulation Time')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out_path + f'magnetization_time.png')


if __name__ == '__main__':
    main()
