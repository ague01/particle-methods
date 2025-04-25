import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib backend for non interactive plotting
mpl.use('Agg')


def plot_csv(file_path, file_name):
    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file '{file_name}': {e}")
        return

    # Check if the CSV has at least two columns
    if data.shape[1] < 2:
        print(f"Error: The CSV file '{file_name}' must have at least two columns.")
        return

    # Plot the data based on the file name
    try:
        if file_name.lower() == "a_temperature_step_size.csv":

            x = data.iloc[:, 0]  # First column as x-axis
            y = data.iloc[:, 1]  # Second column as y-axis
            plt.figure(figsize=(10, 8))
            plt.plot(x, y, marker='o', linestyle='-', label='Equilibrium Temperature')
            plt.axhline(y=0.1111, color='r', linestyle='--', label='Theoretical Value')
            plt.yscale('log')
            plt.xticks(x, rotation=45)
            plt.xlabel('Step Size')
            plt.ylabel(r'Equilibrium Temperature $k_BT$')
            plt.title('Temperature vs Step Size')

        elif file_name.lower() == "b_courette_flow.csv":

            animate(data, file_name, file_path)

        plt.legend()
        plt.grid(True, which='both')
        # Save the plot as a PNG file
        output_folder = os.path.join(os.path.dirname(file_path), 'plots')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to '{output_file}'")

    except Exception as e:
        print(f"Error plotting the data for '{file_name}': {e}")


def animate(df, file_name, file_path):
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle

    fps = 20
    marker_size = 20
    output_video = os.path.join(os.path.dirname(file_path), 'plots',
                                f"{os.path.splitext(file_name)[0]}.mp4")

    time_steps = sorted(df['Time'].unique())

    # Define color map for types 0-3
    particle_colors = {
        0: 'tab:blue',
        1: 'tab:orange',
        2: 'tab:green',
        3: 'tab:red'
    }

    # Map particle types to colors
    df['color'] = df['type'].map(particle_colors)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    # Add walls to the plot
    wall_left = Rectangle((0, 0), 1, 15.5, color='gray', alpha=0.5)
    wall_right = Rectangle((14, 0), 1, 15.5, color='gray', alpha=0.5)
    ax.add_patch(wall_left)
    ax.add_patch(wall_right)
    # Add particles to the plot
    scat = ax.scatter([], [], s=marker_size, c=[], edgecolors='k')
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 15.5)
    ax.set_aspect('equal')
    ax.set_title('DPD Particle Simulation')

    def update(frame):
        step = time_steps[frame]
        frame_data = df[df['Time'] == step]
        positions = frame_data[['X', 'Y']].values
        colors = frame_data['color'].values
        scat.set_offsets(positions)
        scat.set_color(colors)
        ax.set_title(f"Time Step: {step}")
        return scat,

    # === CREATE ANIMATION ===
    ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=1000/fps, blit=True)

    # === SAVE TO MP4 ===
    ani.save(output_video, writer='ffmpeg', fps=fps)
    print(f'Animation saved to {output_video}')


def plot_all_csv_in_out_folder():
    # Define the path to the 'out' folder
    out_folder = os.path.join(os.path.dirname(__file__), 'out')

    # Check if the folder exists
    if not os.path.exists(out_folder):
        print("Error: 'out' folder not found.")
        return

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(out_folder) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the 'out' folder.")
        return

    # Plot each CSV file
    for file_name in csv_files:
        file_path = os.path.join(out_folder, file_name)
        print(f"Plotting '{file_name}'...")
        plot_csv(file_path, file_name)


if __name__ == "__main__":
    plot_all_csv_in_out_folder()
