import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time

# Constants
WIDTH, HEIGHT = 1600, 900
NUM_PARTICLES = 100
MAX_SPEED = 50
TIME_STEP = 0.016  # ~60 FPS
MIN_MASS, MAX_MASS = 1, 5
MIN_DENSITY, MAX_DENSITY = 0.005, 0.01

GRAVITATIONAL_CONSTANT = 0.1
REPULSION_FORCE_CONSTANT = 50
REPULSION_DAMPING_FACTOR = 0.95

# Grid parameters
GRID_SIZE = 100
GRID_WIDTH = WIDTH // GRID_SIZE + 1
GRID_HEIGHT = HEIGHT // GRID_SIZE + 1

# Camera parameters
camera_x, camera_y = WIDTH / 2, HEIGHT / 2
zoom = 1.0
PAN_SPEED = 10
ZOOM_FACTOR = 1.1

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
        self.grid_com = np.zeros((GRID_WIDTH * GRID_HEIGHT, 3))  # x, y, total_mass

    def update_grid(self):
        for cell in self.grid:
            cell.clear()
        self.grid_com.fill(0)
        
        cell_x = (self.positions[:, 0] / GRID_SIZE).astype(int)
        cell_y = (self.positions[:, 1] / GRID_SIZE).astype(int)
        cell_indices = cell_y * GRID_WIDTH + cell_x
        
        for i, cell_index in enumerate(cell_indices):
            self.grid[cell_index].append(i)
            self.grid_com[cell_index, 0] += self.positions[i, 0] * self.masses[i]
            self.grid_com[cell_index, 1] += self.positions[i, 1] * self.masses[i]
            self.grid_com[cell_index, 2] += self.masses[i]
        
        # Finalize center of mass calculations
        mask = self.grid_com[:, 2] > 0
        self.grid_com[mask, :2] /= self.grid_com[mask, 2:3]

    def apply_forces(self):
        self.forces.fill(0)
        for cell_index, cell in enumerate(self.grid):
            if not cell:
                continue
            
            # Pairwise interactions within the cell
            indices = np.array(cell)
            pos = self.positions[indices]
            mass = self.masses[indices]
            radii = self.radii[indices]
            
            dx = pos[:, 0][:, np.newaxis] - pos[:, 0]
            dy = pos[:, 1][:, np.newaxis] - pos[:, 1]
            
            d2 = np.maximum(1e-6, dx**2 + dy**2)  # Increased minimum distance
            d = np.sqrt(d2)
            
            Fg = GRAVITATIONAL_CONSTANT * mass[:, np.newaxis] * mass / d2
            
            overlap = np.maximum(0, radii[:, np.newaxis] + radii - d)
            epsilon = 1e-10  # Small value to prevent log(0)
            Fr = REPULSION_FORCE_CONSTANT * np.log1p(REPULSION_DAMPING_FACTOR * overlap + epsilon)
            
            F = Fg - Fr
            F[d2 < 1e-6] = 0  # Avoid self-interaction
            
            Fx = F * (dx / d)
            Fy = F * (dy / d)
            
            np.add.at(self.forces, (indices, 0), np.sum(Fx, axis=1))
            np.add.at(self.forces, (indices, 1), np.sum(Fy, axis=1))
            
            # Interactions with other cells' centers of mass
            cell_x = cell_index % GRID_WIDTH
            cell_y = cell_index // GRID_WIDTH
            for other_cell_index, com in enumerate(self.grid_com):
                if other_cell_index == cell_index or com[2] == 0:
                    continue
                
                dx = com[0] - pos[:, 0]
                dy = com[1] - pos[:, 1]
                d2 = np.maximum(1e-6, dx**2 + dy**2)
                d = np.sqrt(d2)
                
                Fg = GRAVITATIONAL_CONSTANT * mass * com[2] / d2
                Fx = Fg * (dx / d)
                Fy = Fg * (dy / d)
                
                self.forces[indices, 0] += Fx
                self.forces[indices, 1] += Fy

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

def mouse_button_callback(window, button, action, mods):
    global last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            last_mouse_x, last_mouse_y = None, None

def cursor_position_callback(window, xpos, ypos):
    global camera_x, camera_y, last_mouse_x, last_mouse_y
    if last_mouse_x is not None and last_mouse_y is not None:
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y
        camera_x -= dx / zoom
        camera_y -= dy / zoom
        last_mouse_x, last_mouse_y = xpos, ypos

def scroll_callback(window, xoffset, yoffset):
    global zoom, camera_x, camera_y
    zoom_center_x = WIDTH / 2
    zoom_center_y = HEIGHT / 2
    
    old_zoom = zoom
    zoom *= ZOOM_FACTOR ** yoffset
    zoom = max(0.1, min(zoom, 10.0))  # Limit zoom range
    
    # Adjust camera position to zoom towards the center of the screen
    camera_x += (zoom_center_x - camera_x) * (1 - zoom / old_zoom)
    camera_y += (zoom_center_y - camera_y) * (1 - zoom / old_zoom)

def handle_keyboard_input(window):
    global camera_x, camera_y
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera_x -= PAN_SPEED / zoom
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera_x += PAN_SPEED / zoom
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_y -= PAN_SPEED / zoom
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_y += PAN_SPEED / zoom

def main():
    global last_mouse_x, last_mouse_y

    if not glfw.init():
        return

    window = glfw.create_window(WIDTH, HEIGHT, "Optimized 2D Particle Simulation", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)

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
    last_mouse_x, last_mouse_y = None, None

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        handle_keyboard_input(window)

        simulation.update_grid()
        simulation.apply_forces()
        simulation.update(TIME_STEP)

        glClear(GL_COLOR_BUFFER_BIT)

        # Apply camera transformations
        glLoadIdentity()
        glTranslatef(WIDTH/2, HEIGHT/2, 0)
        glScalef(zoom, zoom, 1)
        glTranslatef(-camera_x, -camera_y, 0)

        simulation.draw()

        glfw.swap_buffers(window)
        glfw.poll_events()

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"FPS: {1.0 / delta_time:.2f}, Particles: {NUM_PARTICLES}")

    glfw.terminate()

if __name__ == "__main__":
    main()