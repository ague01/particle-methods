# Dissipative Particle Dynamics (DPD)

This repository contains a Python implementation of the Dissipative Particle Dynamics (DPD) method following the Groot and Warren model. The code is designed to simulate the dynamics of a system of particles interacting through soft potentials, with the ability to include hydrodynamic interactions.

## Running the code

To run the code, you need to have Python 3.x installed along with the required libraries. You can install the required libraries using pipenv from the `particle-methods` directory:

```bash
cd particle-methods
pipenv install
```

Then, you can run the simulation using the following command:

```bash
cd dpd-groot
pipenv run python dpd_simulation.py
pipenv run python dpd_plotter.py
```

## Results

### Equilibrium temperature as a function of time step

The variation of the equilibrium temperature with respect to the time step is shown in Figure 1. The red line indicates the theoretical value of the temperature, i.e. $k_BT = \frac{\sigma^2}{2\gamma} = 0.1111$, while the blue line indicates the value obtained from the simulation.

<figure>
  <center>
 <img src="./out/plots/a_temperature_step_size.png"
   alt="Temperature as function of time step"
   width=70%>
  </center>
 <figcaption><em> Figure 1: Equilibrium temperature as a function of time step. The red line indicates the theoretical value of the temperature, while the blue line indicates the value obtained from the simulation.</em></figcaption>
</figure>

The results suggest that the system reaches a stable temperature for small time steps around $10^{-1}$ while larger time steps lead to deviations from the expected value.

### Courette flow with string molecules

The simulation of a Couette flow with string particles is shown in Figure 2. The system consists of a fluid confined between two walls, moving in opposite directions (left upwards, right downwards). 42 string molecules are used, each composed of 7 particles of type A-A-B-B-B-B-B. Type B particles have a low conservative force coefficient when interacting with same type particles. While Type A particles have a high conservative force coefficient when interacting with same type particles. The system is initialized with a density of 4, the simulation runs for 1000 steps, with a time step of 0.01.

<figure>
  <center>
    <video width="70%" controls>
      <source src="out/plots/b_courette_flow.mp4" type="video/mp4">
    </video>
  </center>
 <figcaption><em> Figure 2: Couette flow with string particles. The system consists of a fluid confined between two walls, moving in opposite directions. The video shows the evolution of the system over time. Only 100 fluid particles are represented to avoid confusion. </em></figcaption>
</figure>

The motion of particles is influenced by the moving walls, the bonds between particles in the same string molecule, and the configuration of conservative force coefficients.

We observe that particles are dragged by the moving walls movement in the same direction, while the spring forces between particles in the same string molecule lead to the usual string structure.

Type B particles tends to cluster together, while type A particles tend to be more dispersed, due to the low conservative force coefficient between type B particles, which leads to a weaker interaction between them. The high conservative force coefficient between type A particles leads to a stronger interaction, pushing them away.

This lead to star-shaped clusters of string particles, that follows the direction of the nearest wall, while in the center they almost compensate eachother. Clusters are approximately evenly distributed in the area between the two walls and results in a circular flow of the fluid and molecules. Type B particles tend to cluster together as they come closer one to another, creating larger structures of string particles with the type A end pointing outwards.

### Poiseuille flow with ring molecules

Poiseuille flow is implemented with fixed static walls and a constant upward force $F=0.3$. The system has 10 ring molecules of 9 particles each, with fluid density 4. The simulation runs for 7500 steps, with a time step of 0.01.

<figure>
  <center>
    <video width="70%" controls>
      <source src="out/plots/c_poiseouille_flow.mp4" type="video/mp4">
    </video>
  </center>
 <figcaption><em> Figure 3: Poiseuille flow with ring particles. The system consists of a fluid confined between two static walls. A constant upward force is applied. The video shows the evolution of the system over time. Only 150 fluid particles are represented to avoid confusion.</em></figcaption>
</figure>

The motion of particles is influenced by the upward force, the bonds between particles in the same ring molecule, and the closeness to the walls of the particle. Particles and molecules move upwards, with higher velocity projection along $y$-axis for particles in the middle between walls. As particles are near a wall, their velocity tend to be lower, till very small values.
Ring molecules near walls show a rotational motion around their centre of mass due to the interaction with the wall, similar to friction. Particles near walls show a turbulent motion with low acceleration. Molecules in the centre of the flow does not show rotational motion, but they show an higher acceleration for the upward force. Same for fluid particles in the centre of the flow, which show a higher velocity projection along $y$-axis.
