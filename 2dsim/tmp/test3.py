import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.optimize import approx_fprime

# Constants
WIDTH, HEIGHT = 200, 200
NUM_CIRCLES = 3
MAX_SPEED = 5
TIME_STEP = 0.0001  # Simulation integrator time step
DISPLAY_INTERVAL = 0.1  # Display update interval
MIN_MASS = 40
MAX_MASS = 50
DENSITY = 0.5
PAN_STEP = 10
ZOOM_STEP = 1.2

# Gravitational constant and repulsion constants
GRAVITATIONAL_CONSTANT = 100  # Adjusted for the simulation scale
REPULSION_FORCE_CONSTANT = 5000  # Adjusted for the overall strength of repulsion
DAMPING_FACTOR = 1  # Adjusted for smoothing effect

# Circle class
class Circle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mass = random.uniform(MIN_MASS, MAX_MASS)
        self.radius = np.sqrt(self.mass / DENSITY / np.pi)
        self.vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.fx = 0
        self.fy = 0

    def apply_force(self, fx, fy):
        self.fx += fx
        self.fy += fy

    def update_position(self, dt):
        self.x += self.vx * dt + 0.5 * (self.fx / self.mass) * dt**2
        self.y += self.vy * dt + 0.5 * (self.fy / self.mass) * dt**2

    def update_velocity(self, dt):
        self.vx += (self.fx / self.mass) * dt
        self.vy += (self.fy / self.mass) * dt

    def reset_forces(self):
        self.fx = 0
        self.fy = 0

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)

# Potential Energy Functions
def gravitational_potential_energy(os, gravitational_constant):
    pe = 0
    for i in range(len(os)):
        for j in range(i + 1, len(os)):
            c1 = os[i]
            c2 = os[j]
            dx = c2.x - c1.x
            dy = c2.y - c1.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance == 0:
                continue
            pe += -gravitational_constant * (c1.mass * c2.mass) / distance
    return pe

def repulsive_potential_energy(os, repulsion_constant, damping_factor):
    pe = 0
    for i in range(len(os)):
        for j in range(i + 1, len(os)):
            c1 = os[i]
            c2 = os[j]
            dx = c2.x - c1.x
            dy = c2.y - c1.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance == 0:
                continue
            overlap = c1.radius + c2.radius - distance
            if overlap > 0:
                pe += 0.5 * repulsion_constant * np.log(1 + damping_factor * overlap)**2
    return pe

def combined_potential_energy(os, potential_functions):
    total_energy = 0
    for func in potential_functions:
        total_energy += func(os)
    return total_energy

def compute_forces(os, combined_energy_func):
    epsilon = np.sqrt(np.finfo(float).eps)
    forces = []

    def potential_energy_wrapper(pos, os, index, original_pos):
        os[index].x, os[index].y = pos
        energy = combined_energy_func(os)
        os[index].x, os[index].y = original_pos
        return energy

    for i, c1 in enumerate(os):
        original_pos = np.array([c1.x, c1.y])
        pos = original_pos.copy()
        grad = approx_fprime(pos, potential_energy_wrapper, epsilon, os, i, original_pos)
        forces.append(-grad)

    return forces

# Integrator Function
def verlet_integrator(os, forces, dt):
    for o, f in zip(os, forces):
        o.apply_force(f[0], f[1])
        o.update_position(dt)
    for o, f in zip(os, forces):
        o.update_velocity(dt)
        o.reset_forces()

# Initialize circles
os = []
for _ in range(NUM_CIRCLES):
    o = Circle(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
    os.append(o)

# Define potential energy functions
potential_functions = [
    lambda os: gravitational_potential_energy(os, GRAVITATIONAL_CONSTANT),
    lambda os: repulsive_potential_energy(os, REPULSION_FORCE_CONSTANT, DAMPING_FACTOR)
]

# Visualization function
def plot_os(os, ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.clear()
    for o in os:
        o_patch = plt.Circle((o.x, o.y), o.radius, color='blue', alpha=0.5)
        ax.add_patch(o_patch)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Set up the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)

def on_key(event):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    if event.key == 'up':
        ax.set_ylim(ylim[0] + PAN_STEP, ylim[1] + PAN_STEP)
    elif event.key == 'down':
        ax.set_ylim(ylim[0] - PAN_STEP, ylim[1] - PAN_STEP)
    elif event.key == 'left':
        ax.set_xlim(xlim[0] - PAN_STEP, xlim[1] - PAN_STEP)
    elif event.key == 'right':
        ax.set_xlim(xlim[0] + PAN_STEP, xlim[1] + PAN_STEP)
    elif event.key == '+':
        ax.set_xlim(xlim[0] + x_range * (1 - 1 / ZOOM_STEP), xlim[1] - x_range * (1 - 1 / ZOOM_STEP))
        ax.set_ylim(ylim[0] + y_range * (1 - 1 / ZOOM_STEP), ylim[1] - y_range * (1 - 1 / ZOOM_STEP))
    elif event.key == '-':
        ax.set_xlim(xlim[0] - x_range * (1 - 1 / ZOOM_STEP), xlim[1] + x_range * (1 - 1 / ZOOM_STEP))
        ax.set_ylim(ylim[0] - y_range * (1 - 1 / ZOOM_STEP), ylim[1] + y_range * (1 - 1 / ZOOM_STEP))
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key)

# Simulation loop
simulation_time = 0
next_display_time = 0
last_time = time.time()

while True:
    current_time = time.time()
    elapsed_time = current_time - last_time
    last_time = current_time

    # Compute forces from combined potential energy
    forces = compute_forces(os, lambda os: combined_potential_energy(os, potential_functions))

    # Integrate the system with fixed small time step
    steps = int(elapsed_time / TIME_STEP)
    for _ in range(steps):
        verlet_integrator(os, forces, TIME_STEP)
        simulation_time += TIME_STEP

    # Update the display at the specified interval
    if simulation_time >= next_display_time:
        plot_os(os, ax)
        plt.pause(0.001)
        next_display_time += DISPLAY_INTERVAL
