import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time

# Constants
WIDTH, HEIGHT = 1600, 900
NUM_PARTICLES = 50000
MAX_SPEED = 50
TIME_STEP = 0.016  # ~60 FPS
MIN_MASS, MAX_MASS = 1, 5
MIN_DENSITY, MAX_DENSITY = 0.005, 0.01

GRAVITATIONAL_CONSTANT = 0.1
REPULSION_FORCE_CONSTANT = 50
REPULSION_DAMPING_FACTOR = 0.95

# Grid parameters
GRID_SIZE = 50
GRID_WIDTH = WIDTH // GRID_SIZE + 1
GRID_HEIGHT = HEIGHT // GRID_SIZE + 1

class Simulation:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.positions = np.random.rand(num_particles, 2) * [WIDTH, HEIGHT]
        self.velocities = (np.random.rand(num_particles, 2) - 0.5) * MAX_SPEED
        self.masses = np.random.uniform(MIN_MASS, MAX_MASS, num_particles)
        self.densities = np.random.uniform(MIN_DENSITY, MAX_DENSITY, num_particles)
        self.radii = np.sqrt(self.masses / (self.densities * np.pi))
        self.colors = np.random.rand(num_particles, 3).astype(np.float32)
        self.forces = np.zeros((num_particles, 2))

        self.grid = [[] for _ in range(GRID_WIDTH * GRID_HEIGHT)]

    def update_grid(self):
        for cell in self.grid:
            cell.clear()
        cell_x = (self.positions[:, 0] / GRID_SIZE).astype(int)
        cell_y = (self.positions[:, 1] / GRID_SIZE).astype(int)
        cell_indices = cell_y * GRID_WIDTH + cell_x
        for i, cell_index in enumerate(cell_indices):
            self.grid[cell_index].append(i)

    def apply_forces(self):
        self.forces.fill(0)
        for cell_index, cell in enumerate(self.grid):
            if len(cell) < 2:
                continue
            
            indices = np.array(cell)
            pos = self.positions[indices]
            mass = self.masses[indices]
            radii = self.radii[indices]
            
            dx = pos[:, 0][:, np.newaxis] - pos[:, 0]
            dy = pos[:, 1][:, np.newaxis] - pos[:, 1]
            
            d2 = np.maximum(1e-1, dx**2 + dy**2)
            d = np.sqrt(d2)
            
            Fg = GRAVITATIONAL_CONSTANT * mass[:, np.newaxis] * mass / d2
            
            overlap = radii[:, np.newaxis] + radii - d
            Fr = np.where(overlap > 0, REPULSION_FORCE_CONSTANT * np.log(1 + REPULSION_DAMPING_FACTOR * overlap), 0)
            
            F = Fg - Fr
            F[d2 < 1e-6] = 0  # Avoid self-interaction
            
            Fx = F * (dx / d)
            Fy = F * (dy / d)
            
            np.add.at(self.forces, (indices, 0), np.sum(Fx, axis=1))
            np.add.at(self.forces, (indices, 1), np.sum(Fy, axis=1))

    def update(self, dt):
        self.velocities += self.forces / self.masses[:, np.newaxis] * dt
        self.positions += self.velocities * dt
        
        # Wrap around screen
        self.positions %= [WIDTH, HEIGHT]

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(2, GL_FLOAT, 0, self.positions)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        
        glPointSize(2)
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

def check_opengl_acceleration():
    vendor = glGetString(GL_VENDOR).decode('utf-8')
    renderer = glGetString(GL_RENDERER).decode('utf-8')
    version = glGetString(GL_VERSION).decode('utf-8')
    print(f"OpenGL Vendor: {vendor}")
    print(f"OpenGL Renderer: {renderer}")
    print(f"OpenGL Version: {version}")
    return "software" not in renderer.lower()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(WIDTH, HEIGHT, "Optimized 2D Particle Simulation", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    if not check_opengl_acceleration():
        print("Warning: OpenGL acceleration might not be enabled!")

    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    simulation = Simulation(NUM_PARTICLES)

    last_time = glfw.get_time()
    frame_count = 0

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        simulation.update_grid()
        simulation.apply_forces()
        simulation.update(TIME_STEP)

        glClear(GL_COLOR_BUFFER_BIT)
        simulation.draw()
        glfw.swap_buffers(window)
        glfw.poll_events()

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"FPS: {1.0 / delta_time:.2f}, Particles: {NUM_PARTICLES}")

    glfw.terminate()

if __name__ == "__main__":
    main()