import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import approx_fprime

# Constants
WIDTH, HEIGHT = 200, 200
NUM_CIRCLES = 10
MAX_SPEED = 5
TIME_STEP = 0.01  # Integrator time step
DISPLAY_INTERVAL = 0.1  # Display update interval
MIN_MASS = 40
MAX_MASS = 50
DENSITY = 0.01
PAN_STEP = 10
ZOOM_STEP = 1.2

# Gravitational constant and repulsion constants
GRAVITATIONAL_CONSTANT = 1000  # Adjusted for the simulation scale
REPULSION_FORCE_CONSTANT = 5000  # Adjusted for the overall strength of repulsion
DAMPING_FACTOR = 0.9  # Adjusted for smoothing effect

# Base class for force fields
class ForceField:
    def apply_force(self, os):
        epsilon = np.sqrt(np.finfo(float).eps)  # Optimal epsilon for numerical differentiation

        def potential_energy_wrapper(pos, os, index, original_pos):
            # Update the position of the circle
            os[index].x, os[index].y = pos
            # Compute the total potential energy
            energy = self.energy(os)
            # Restore the original position
            os[index].x, os[index].y = original_pos
            return energy

        for i, c1 in enumerate(os):
            original_pos = np.array([c1.x, c1.y])
            pos = original_pos.copy()

            # Compute the gradient (force) using scipy's approx_fprime
            grad = approx_fprime(pos, potential_energy_wrapper, epsilon, os, i, original_pos)

            # Apply the force to the circle
            c1.apply_force(-grad[0], -grad[1])

    def energy(self, os):
        raise NotImplementedError("ForceField is an abstract base class")

# Gravitational force field
class GravitationalForceField(ForceField):
    def __init__(self, gravitational_constant):
        self.gravitational_constant = gravitational_constant

    def apply_force(self, os):
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                distance = np.sqrt(dx**2 + dy**2)
                if distance == 0:
                    continue  # Avoid division by zero

                # Calculate gravitational force analytically
                force_gravity = self.gravitational_constant * (c1.mass * c2.mass) / distance**3
                fx_gravity = force_gravity * dx
                fy_gravity = force_gravity * dy
                c1.apply_force(fx_gravity, fy_gravity)
                c2.apply_force(-fx_gravity, -fy_gravity)

    def energy(self, os):
        pe = 0
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                distance = np.sqrt(dx**2 + dy**2)
                if distance == 0:
                    continue  # Avoid division by zero
                # Calculate gravitational potential energy
                pe += -self.gravitational_constant * (c1.mass * c2.mass) / distance
        return pe

# Repulsive force field for overlapping objects
class RepulsiveForceField(ForceField):
    def __init__(self, repulsion_constant, damping_factor):
        self.repulsion_constant = repulsion_constant
        self.damping_factor = damping_factor

    def apply_force(self, os):
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                distance = np.sqrt(dx**2 + dy**2)
                if distance == 0:
                    continue  # Avoid division by zero

                # Calculate repulsive force analytically
                overlap = c1.radius + c2.radius - distance
                if overlap > 0:
                    force_repulsion = self.repulsion_constant * self.damping_factor * overlap / (1 + self.damping_factor * overlap)
                    fx_repulsion = force_repulsion * dx / distance
                    fy_repulsion = force_repulsion * dy / distance
                    c1.apply_force(-fx_repulsion, -fy_repulsion)
                    c2.apply_force(fx_repulsion, fy_repulsion)

    def energy(self, os):
        pe = 0
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                distance = np.sqrt(dx**2 + dy**2)
                if distance == 0:
                    continue  # Avoid division by zero
                # Calculate repulsive potential energy if there's an overlap
                overlap = c1.radius + c2.radius - distance
                if overlap > 0:
                    pe += 0.5 * self.repulsion_constant * np.log(1 + self.damping_factor * overlap)**2
        return pe

class Entity:
    pass

class AtomicEntity(Entity):
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

    def verlet_step1(self):
        # Update position based on current velocity and acceleration
        self.x += self.vx * TIME_STEP + 0.5 * (self.fx / self.mass) * TIME_STEP**2
        self.y += self.vy * TIME_STEP + 0.5 * (self.fy / self.mass) * TIME_STEP**2

        # Half update to velocity
        self.vx += 0.5 * (self.fx / self.mass) * TIME_STEP
        self.vy += 0.5 * (self.fy / self.mass) * TIME_STEP

    def verlet_step2(self):
        # Update velocity based on new force
        self.vx += 0.5 * (self.fx / self.mass) * TIME_STEP
        self.vy += 0.5 * (self.fy / self.mass) * TIME_STEP

        # Reset forces
        self.fx = 0
        self.fy = 0

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)

def create_objects(n):
    os = []
    for _ in range(n):
        o = AtomicEntity(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        os.append(o)
    return os

# Calculate total kinetic energy
def total_kinetic_energy(os):
    return sum(o.kinetic_energy() for o in os)

# Calculate total potential energy
def total_potential_energy(os, force_fields):
    return sum(field.energy(os) for field in force_fields)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)

# Visualization function
def plot_objects(os):
    # Store current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.clear()
    for o in os:
        o_patch = plt.Circle((o.x, o.y), o.radius, color='blue', alpha=0.5)
        ax.add_patch(o_patch)
    
    # Restore limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# Key event handler for panning and zooming
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

# Initialize force fields
def initialize_force_fields():
    return [
        GravitationalForceField(GRAVITATIONAL_CONSTANT),
        RepulsiveForceField(REPULSION_FORCE_CONSTANT, DAMPING_FACTOR)
    ]

# Simulation loop
def simulation_loop(os, force_fields):
    simulation_time = 0
    next_display_time = 0
    i = 1000

    while True:
        # Integrate the system
        for o in os:
            o.verlet_step1()
        for field in force_fields:
            field.apply_force(os)
        for o in os:
            o.verlet_step2()

        # Update the display at the specified interval
        simulation_time += TIME_STEP
        if simulation_time >= next_display_time:
            i += 1
            plot_objects(os)
            plt.pause(0.001)
            if i > 100:
                ke = total_kinetic_energy(os)
                pe = total_potential_energy(os, force_fields)
                te = ke + pe
                print(f"KE: {ke:.2f}, PE: {pe:.2f}, E: {te:.2f}")
                i = 0
            next_display_time += DISPLAY_INTERVAL

# Main function to run the simulation
def main():
    os = create_objects(NUM_CIRCLES)
    force_fields = initialize_force_fields()
    simulation_loop(os, force_fields)

if __name__ == "__main__":
    main()
