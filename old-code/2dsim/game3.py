import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import time
import threading

# Constants
WIDTH, HEIGHT = 800, 600
NUM_CIRCLES = 100
MAX_SPEED = 5
TIME_STEP = 0.05  # Integrator time step
DISPLAY_INTERVAL = 0.1 # Display update interval
MIN_MASS = 40
MAX_MASS = 50
MIN_DENSITY = 0.05
MAX_DENSITY = 0.1

# Gravitational constant and repulsion constants
GRAVITATIONAL_CONSTANT = 1000  # Adjusted for the simulation scale
REPULSION_FORCE_CONSTANT = 10000  # Adjusted for the overall strength of repulsion
REPULSION_DAMPING_FACTOR = 1.0  # Adjusted for smoothing effect

# Base class for interactions
class PotentialField:
    """
    Base class for potential fields.
    """
    def apply_force(self, os, eps=1e-2):
        for i in range(len(os)):
            c1 = os[i]

            # Calculate force in x direction
            c1.x += eps
            U1 = self.energy(os)
            c1.x -= 2 * eps
            U2 = self.energy(os)
            c1.x += eps  # Reset position
            Fx = -(U1 - U2) / (2 * eps)

            # Calculate force in y direction
            c1.y += eps
            U1 = self.energy(os)
            c1.y -= 2 * eps
            U2 = self.energy(os)
            c1.y += eps  # Reset position
            Fy = -(U1 - U2) / (2 * eps)

            c1.apply_force(Fx, Fy)

    def energy(self, os):
        raise NotImplementedError("PotentialField is an abstract base class")

# Gravitational force field
class GravField(PotentialField):
    """
    Field that applies a gravitational force between objects.
    """
    def __init__(self, G, min_d=1e-1):
        self.G = G
        self.min_d = min_d

    def apply_force(self, os):
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                d = max(self.min_d, np.sqrt(dx**2 + dy**2))
                F = self.G * (c1.mass * c2.mass) / d**2
                Fx = F * dx / d
                Fy = F * dy / d
                c1.apply_force(Fx, Fy)
                c2.apply_force(-Fx, -Fy)

    def energy(self, os):
        pe = 0
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                d = np.sqrt(dx**2 + dy**2)
                d = max(self.min_d, d)
                pe += -self.G * (c1.mass * c2.mass) / d
        return pe

class ExclusionField(PotentialField):
    """
    Field that applies a repulsive force between objects that overlap.
    """
    def __init__(self, k, damp, min_d=1e-1):
        """
        k: Repulsion constant
        damp: Damping factor for the repulsion
        min_d: Minimum distance to avoid division by zero
        """
        self.k = k
        self.damp = damp
        self.min_d = min_d

    def apply_force(self, os):
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                d = max(self.min_d, np.sqrt(dx**2 + dy**2))
                overlap = c1.radius + c2.radius - d
                if overlap > 0:
                    F = self.k * np.log(1 + self.damp * overlap)
                    Fx = F * dx / d
                    Fy = F * dy / d
                    c1.apply_force(-Fx, -Fy)        
                    c2.apply_force(Fx, Fy)

    def energy(self, os):
        pe = 0
        for i in range(len(os)):
            for j in range(i + 1, len(os)):
                c1 = os[i]
                c2 = os[j]
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                d = max(self.min_d, np.sqrt(dx**2 + dy**2))
                overlap = c1.radius + c2.radius - d
                if overlap > 0:
                    pe += 0.5 * self.k * np.log(1 + self.damp * overlap)**2
        return pe

class Entity:
    def __init__(self, x, y, vx=0, vy=0, mass=1, density=0.1):
        self.x = x
        self.y = y
        self.mass = mass

        self.radius = np.sqrt(self.mass / density / np.pi)
        self.density = density
        self.vx = vx
        self.vy = vy
        self.fx = 0
        self.fy = 0
        self.color = (random.random(), random.random(), random.random()) 

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
        self.vx += 0.5 * (self.fx / self.mass) * TIME_STEP
        self.vy += 0.5 * (self.fy / self.mass) * TIME_STEP

        self.fx = 0
        self.fy = 0

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)
    
    def draw(self):
        segments = 100
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(*self.color)  # Set color
        glVertex2f(self.x, self.y)
        for i in range(segments + 1):
            theta = 2.0 * np.pi * i / segments
            dx = self.radius * np.cos(theta)
            dy = self.radius * np.sin(theta)
            glVertex2f(self.x + dx, self.y + dy)
        glEnd()

        # Draw a white border for better visibility
        glColor3f(1.0, 1.0, 1.0)  # White color
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            dx = self.radius * np.cos(theta)
            dy = self.radius * np.sin(theta)
            glVertex2f(self.x + dx, self.y + dy)
        glEnd()        

# Create circles
os = []
for _ in range(NUM_CIRCLES):
    speed = random.uniform(0, MAX_SPEED)
    angle = random.uniform(0, 2 * np.pi)
    o = Entity(x=random.uniform(0, WIDTH),
               y=random.uniform(0, HEIGHT),
               vx=speed * np.cos(angle),
               vy=speed * np.sin(angle),
               mass=random.uniform(MIN_MASS, MAX_MASS),
               density=random.uniform(MIN_DENSITY, MAX_DENSITY))
    os.append(o)

# Calculate total kinetic energy
def total_kinetic_energy(os):
    return sum(o.kinetic_energy() for o in os)

# Calculate total potential energy
def total_potential_energy(os, interactions):
    return sum(field.energy(os) for field in interactions)

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(WIDTH, HEIGHT, "2D Simulation", None, None)

if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.set_window_pos(window, 400, 200)
glfw.make_context_current(window)

# Set up viewport
glViewport(0, 0, WIDTH, HEIGHT)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

potential_fields = [
    GravField(GRAVITATIONAL_CONSTANT, 1.0),
    ExclusionField(REPULSION_FORCE_CONSTANT, REPULSION_DAMPING_FACTOR, 1.0)
]

def integrate_system(os, potential_fields):
    for o in os:
        o.verlet_step1()
    for field in potential_fields:
        field.apply_force(os)
    for o in os:
        o.verlet_step2()

def physics_loop():
    while not glfw.window_should_close(window):       
        start_time = time.time()
        integrate_system(os, potential_fields)
        end_time = time.time()
        time.sleep(max(0, TIME_STEP - (end_time - start_time)))
        print(f"Physics loop time: {end_time - start_time:.5f} seconds")

def render_system(os):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Draw a white border around the window
    glColor3f(1.0, 1.0, 1.0)  # White color
    glBegin(GL_LINE_LOOP)
    glVertex2f(0, 0)
    glVertex2f(WIDTH, 0)
    glVertex2f(WIDTH, HEIGHT)
    glVertex2f(0, HEIGHT)
    glEnd()
    
    for o in os:
        o.draw()
        
    # Print the position of the first circle
    if os:
        print(f"First circle position: ({os[0].x:.2f}, {os[0].y:.2f}), radius: {os[0].radius:.2f}")
    glfw.swap_buffers(window)

def display_loop():
    last_display_time = time.time()
    frame_count = 0
    while not glfw.window_should_close(window):
        current_time = time.time()
        if current_time - last_display_time >= DISPLAY_INTERVAL:
            glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color to black for better contrast

            render_system(os)

            error = glGetError()
            if error != GL_NO_ERROR:
                print("OpenGL error:", gluErrorString(error))

            actual_display_interval = current_time - last_display_time
            last_display_time = current_time
            frame_count += 1
            if frame_count % 10 == 0:  # Print every 10 frames
                print(f"Frame {frame_count}, Actual display interval: {actual_display_interval:.5f} seconds")
        glfw.poll_events()

# Run physics and display loops in separate threads
physics_thread = threading.Thread(target=physics_loop)
display_thread = threading.Thread(target=display_loop)

physics_thread.start()
display_thread.start()

# Main loop to keep the window open
while not glfw.window_should_close(window):
    glfw.poll_events()

physics_thread.join()
display_thread.join()

glfw.terminate()