# Monte Carlo simulation for Ising Model

## 1. Derivation of $\Delta E$

Given the Hamiltonian for the Ising model:

```math
E = H(\{\sigma\}) = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j
```

where:

- $\sigma_i \in {\plusmn 1}$ is the spin at site $i$
- $J$ is the coupling constant
- $\langle i, j \rangle$ means the sum runs over all pairs of nearest neighbors (up, down, left, right)

The energy difference:

```math
\Delta E = E(X) - E(Y)
```

where $E(Y)$ is the energy of the system after flipping a single spin $\sigma_i$ to $-\sigma_i$, can be derived as follows.

---

The contribution to the total energy from spin $\sigma_i$ interacting with its neighbors $\sigma_j$ is:

```math
E_i = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j = -J \sigma_i \sum_{\langle i,j \rangle} \sigma_j = -J \sigma_i h_i
```

After flipping $\sigma_i$ to $-\sigma_i$, its contribution becomes:

```math
E_i' = -J (-\sigma_i) \sum_{\langle i,j \rangle} \sigma_j = J \sigma_i h_i
```

The change in energy due to flipping $\sigma_i$ is the difference between the new and the old contribution:

```math
\begin{aligned}
\Delta E & = E_i' - E_i = \left( J \sigma_i h_i \right) - \left( -J \sigma_i h_i \right) \\
& =  J \sigma_i h_i + J \sigma_i h_i = 2J \sigma_i h_i
\end{aligned}
```

concluding that the energy difference caused by flipping spin $\sigma_i$ is:

```math
\boxed{ \Delta E = 2J \sigma_i h_i }
```

where:

- $h_i = \sum_{\langle i,j \rangle} \sigma_j$: the sum of the four nearest neighbor spins around $i$
- $J$: the coupling constant that describes the strength of interaction between neighboring spins
