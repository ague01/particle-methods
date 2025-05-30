# Script that plots the data from the csv files and saves them as png files
# Takes the first column as the x-axis and the rest as the y-axis
# The labels are inferred from the column names on the first row after comments
from re import L
import sys
import os
from os import path
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set matplotlib backend for non interactive plotting
mpl.use('Agg')

# If no filename passed, exit
if len(sys.argv) < 2:
	print("Please pass the filename as an argument.")
	print("Only files in the output folder can be plotted.")
	exit()

# Find the path of CSV file
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "output", sys.argv[1]))
savepath = path.abspath(path.join(basepath, "..", "plots"))

# Read the CSV file using pandas
data = pd.read_csv(filepath, comment='#', index_col=0)

# Set the font size
sns.set_theme(font_scale=1.1, style='darkgrid')

# Compose the title from the comments
title = ""
description = "("
with open(filepath, 'r') as f:
	title = f.readline()[1:].strip().replace('_', ' ').capitalize()
	for line in f:
		if line[0] == '#':
			description += line[1:].strip() + ", "
	description = description[:-2] + ")"


ax = None

# ====== Plot for the abc_optimize.csv ======
if sys.argv[1] == "optimize_abc_1.csv":
	fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8), dpi=600)
	print(data.info())

	# Value plot
	ax = sns.lineplot(ax=axes[0], data=(data[['value']] - 24.30620)/24.30620, linestyle='-.', linewidth=2, legend=True)
	ax.set_yscale('linear')
	ax.set_ylabel("Relative Error")
	ax.set_xlabel("Iterations")
	ax.set_xlim(left=-5, right=100)

	# Feasible particles plot as twin of the constraint violation plot
	ax = sns.lineplot(ax=axes[1].twinx(), data=data[['feasible_particles']], linestyle='-.', linewidth=2, markersize=8, palette=['g',], legend=True)
	ax.set_ylabel("Feasible Particles")
	ax.set_xlabel("Iterations")

	# Constraint violation plot
	ax = sns.lineplot(ax=axes[1], data=data[['violation']], linestyle='-.', linewidth=2, palette=['r',], legend=True)
	ax.set_ylabel("Constraint Violation")
	ax.set_xlabel("Iterations")
	ax.set_ylim(bottom=-3)

	# Set axis where to put the title
	ax = axes[0]



# ====== Plot for the abc_time_numparticles.csv ======
elif sys.argv[1] == "time_numparticles_abc_8.csv":
	plt.figure(figsize=(12, 10))
	ax = sns.lineplot(data=data.Serial_time, color='blue', linewidth=3)
	sns.lineplot(data=data.Parallel_time, color='red', ax=ax, linewidth=3)
	ax2 = ax.twinx()
	sns.lineplot(data=data.Speedup, ax=ax2, color='green', linewidth=3)
	ax.legend(handles=[Line2D([], [], marker='_', color='blue', label='Serial'),
					   Line2D([], [], marker='_', color='red',  label='Parallel'),
					   Line2D([], [], marker='_', color='green',  label='Speedup')],
			fontsize=14)
	ax.set_ylabel("Time (ms)")
	ax2.set_ylabel("Speedup")
	ax.set_xlabel("Number of particles")



# ====== If the csv is not recognized ======
else:
	print("File not supported.")
	exit()

# Set description below title
ax.text(x=0.5, y=1.1, s=title, fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.0, s=description, fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)


# Save the plot in the output directory
plt.savefig(os.path.join(savepath, sys.argv[1][:-4] + ".png"))
