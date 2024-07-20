import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import time

# Constants
WIDTH, HEIGHT = 800, 600
NUM_CIRCLES = 200  # Reduced for better performance with actual circles
MAX_SPEED = 5
TIME_STEP = 0.016  # ~60 FPS
MIN_MASS, MAX_MASS = 1, 5
MIN_DENSITY, MAX_DENSITY = 0.005, 0.01

GRAVITATIONAL_CONSTANT = 250
REPULSION_FORCE_CONSTANT = 2000
REPULSION_DAMPING_FACTOR = 0.95

class Entity:
    def __init__(self, x, y, vx=0, vy=0, mass=1, density=0.1):
        self.x, self.y = x, y
        self.mass = mass
        self.radius = np.sqrt(self.mass / (density * np.pi))
        self.vx, self.vy = vx, vy
        self.fx, self.fy = 0, 0
        self.color = np.random.rand(3).astype(np.float32)

    def draw(self):
        num_segments = 8  # Adjust for quality vs. performance
        glColor3fv(self.color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(self.x, self.y)
        for i in range(num_segments + 1):
            theta = 2.0 * np.pi * i / num_segments
            dx = self.radius * np.cos(theta)
            dy = self.radius * np.sin(theta)
            glVertex2f(self.x + dx, self.y + dy)
        glEnd()

def create_entities(num):
    return [Entity(
        x=random.uniform(0, WIDTH),
        y=random.uniform(0, HEIGHT),
        vx=random.uniform(-MAX_SPEED, MAX_SPEED),
        vy=random.uniform(-MAX_SPEED, MAX_SPEED),
        mass=random.uniform(MIN_MASS, MAX_MASS),
        density=random.uniform(MIN_DENSITY, MAX_DENSITY)
    ) for _ in range(num)]

def apply_forces(entities):
    n = len(entities)
    for i in range(n):
        for j in range(i+1, n):
            e1, e2 = entities[i], entities[j]
            dx, dy = e2.x - e1.x, e2.y - e1.y
            d2 = max(1e-1, dx*dx + dy*dy)
            d = np.sqrt(d2)
            Fg = GRAVITATIONAL_CONSTANT * e1.mass * e2.mass / d2 / d
            e1.fx += Fg * dx
            e1.fy += Fg * dy
            e2.fx -= Fg * dx
            e2.fy -= Fg * dy
            
            # Repulsion
            overlap = e1.radius + e2.radius - d
            if overlap > 0:
                Fr = REPULSION_FORCE_CONSTANT * np.log(1 + REPULSION_DAMPING_FACTOR * overlap) / d
                e1.fx -= Fr * dx
                e1.fy -= Fr * dy
                e2.fx += Fr * dx
                e2.fy += Fr * dy           

def update_entities(entities, dt):
    for e in entities:
        e.vx += e.fx / e.mass * dt
        e.vy += e.fy / e.mass * dt
        e.x += e.vx * dt
        e.y += e.vy * dt
        e.fx, e.fy = 0, 0
        
        # Wrap around screen
        #e.x %= WIDTH
        #e.y %= HEIGHT

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
        raise Exception("GLFW can't be initialized")

    window = glfw.create_window(WIDTH, HEIGHT, "2D Simulation with Circles", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.set_window_pos(window, 400, 200)
    glfw.make_context_current(window)

    if not check_opengl_acceleration():
        print("Warning: OpenGL acceleration might not be enabled!")

    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    entities = create_entities(NUM_CIRCLES)

    last_time = glfw.get_time()
    frame_count = 0
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        apply_forces(entities)
        update_entities(entities, TIME_STEP)

        glClear(GL_COLOR_BUFFER_BIT)

        for entity in entities:
            entity.draw()

        glfw.swap_buffers(window)
        glfw.poll_events()

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"FPS: {1.0 / delta_time:.2f}")

    glfw.terminate()

if __name__ == "__main__":
    main()