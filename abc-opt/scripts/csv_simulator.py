import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path


# If no filename passed, exit
if len(sys.argv) < 2:
	print("Please pass the filename as an argument.")
	print("Only files in the output folder can be plotted.")
	exit()

# Find the path of CSV file
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "output", sys.argv[1]))

# Read the CSV file using pandas
df = pd.read_csv(filepath, comment='#')
df['isbest'] = df['isbest'].astype('bool')

# Create figure and axis
fig, ax = plt.subplots()

# Define the Gomez-Levy function with constraints
def ff(x, y):
    if(-np.sin(4 * np.pi * x) + 2 * np.sin(2 * np.pi * y)**2 > 1.5):
        return np.nan
    ret = 0.0
    x2 = x * x
    y2 = y * y
    ret += 4.0 * x2 - 2.1 * x2 * x2 + (1.0 / 3) * x2 * x2 * x2
    ret += x * y
    ret += 4 * y2 * y2 - 4 * y2
    return ret
f = np.vectorize(ff)

x_min = -3.
x_max = 3.
y_min = -3.
y_max = 3.
x_best = 0.089842015605160238656
y_best = -0.71265640151135767333
best = f(x_best, y_best)

# Generate grid data for contour plot
x_grid = np.linspace(x_min - 1, x_max + 1, 1000)
y_grid = np.linspace(y_min - 1, y_max + 1, 1000)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
z_mesh = f(x_mesh, y_mesh)

unique_iters = df['iter'].unique()

def animate(i):
    ax.clear()
    iter_num = unique_iters[i]
    iter_data = df[df['iter'] == iter_num]
    best_point = iter_data[iter_data['isbest'] == True]

    # Plot contour
    contour = ax.contourf(x_mesh, y_mesh, z_mesh, levels=np.linspace(best-0.2, f(x_min,y_min), 1000), cmap='viridis', alpha=0.6)
    # Plot best point
    ax.scatter(x_best, y_best, c='red', marker='+', label='Real optimium')

    ax.scatter(iter_data['x0'], iter_data['x1'], c='blue', label='Points')
    ax.scatter(best_point['x0'], best_point['x1'], c='red', label='Best Point')

    ax.set_title(f'Particle behaviour ABC, iter {iter_num}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()


# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(unique_iters), interval=200, repeat=False) # type: ignore

# Save animation as mp4
ani.save(path.join(basepath, "..", "output", sys.argv[1][:-4] + ".mp4"), writer='ffmpeg')
print(f"Animation saved as {sys.argv[1][:-4]}.mp4")