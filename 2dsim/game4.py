import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import time

# Constants
WIDTH, HEIGHT = 800, 600
NUM_CIRCLES = 100
MAX_SPEED = 5
TIME_STEP = 0.025
MIN_MASS, MAX_MASS = 10, 20
MIN_DENSITY, MAX_DENSITY = 0.05, 0.1

GRAVITATIONAL_CONSTANT = 1000
REPULSION_FORCE_CONSTANT = 10000
REPULSION_DAMPING_FACTOR = 1.0

class PotentialField:
    def apply_force(self, entities):
        pass

class GravField(PotentialField):
    def __init__(self, G, min_d=1e-1):
        self.G, self.min_d = G, min_d

    def apply_force(self, entities):
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                c1, c2 = entities[i], entities[j]
                dx, dy = c2.x - c1.x, c2.y - c1.y
                d = max(self.min_d, np.sqrt(dx**2 + dy**2))
                F = self.G * (c1.mass * c2.mass) / d**2
                Fx, Fy = F * dx / d, F * dy / d
                c1.apply_force(Fx, Fy)
                c2.apply_force(-Fx, -Fy)

class ExclusionField(PotentialField):
    def __init__(self, k, damp, min_d=1e-1):
        self.k, self.damp, self.min_d = k, damp, min_d

    def apply_force(self, entities):
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                c1, c2 = entities[i], entities[j]
                dx, dy = c2.x - c1.x, c2.y - c1.y
                d = max(self.min_d, np.sqrt(dx**2 + dy**2))
                overlap = c1.radius + c2.radius - d
                if overlap > 0:
                    F = self.k * np.log(1 + self.damp * overlap)
                    Fx, Fy = F * dx / d, F * dy / d
                    c1.apply_force(-Fx, -Fy)        
                    c2.apply_force(Fx, Fy)

class Entity:
    def __init__(self, x, y, vx=0, vy=0, mass=1, density=0.1):
        self.x, self.y = x, y
        self.mass = mass
        self.radius = np.sqrt(self.mass / (density * np.pi))
        self.vx, self.vy = vx, vy
        self.fx, self.fy = 0, 0
        self.color = (random.random(), random.random(), random.random())

    def apply_force(self, fx, fy):
        self.fx += fx
        self.fy += fy

    def update(self, dt):
        self.vx += self.fx / self.mass * dt
        self.vy += self.fy / self.mass * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.fx, self.fy = 0, 0

def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_program(vertex_source, fragment_source):
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program))
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

vertex_shader = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in vec3 aColor;

out vec3 fragColor;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = aRadius * 2.0;
    fragColor = aColor;
}
"""

fragment_shader = """
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) {
        discard;
    }
    FragColor = vec4(fragColor, 1.0);
}
"""

# Initialize GLFW and create window
if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(WIDTH, HEIGHT, "2D Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.set_window_pos(window, 400, 200)
glfw.make_context_current(window)

# Check OpenGL version and renderer
print("OpenGL Version:", glGetString(GL_VERSION).decode())
print("OpenGL Renderer:", glGetString(GL_RENDERER).decode())

# Create and use shader program
program = create_program(vertex_shader, fragment_shader)
glUseProgram(program)

# Set up projection matrix
projection = np.array([
    [2/WIDTH, 0, 0, -1],
    [0, -2/HEIGHT, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)
glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, projection)

# Create entities
entities = [
    Entity(
        x=random.uniform(0, WIDTH),
        y=random.uniform(0, HEIGHT),
        vx=random.uniform(-MAX_SPEED, MAX_SPEED),
        vy=random.uniform(-MAX_SPEED, MAX_SPEED),
        mass=random.uniform(MIN_MASS, MAX_MASS),
        density=random.uniform(MIN_DENSITY, MAX_DENSITY)
    )
    for _ in range(NUM_CIRCLES)
]

# Create VBO and VAO
vbo = glGenBuffers(1)
vao = glGenVertexArrays(1)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)

# Set up vertex attributes
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(8))
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

# Create potential fields
potential_fields = [
    GravField(GRAVITATIONAL_CONSTANT, 1.0),
    ExclusionField(REPULSION_FORCE_CONSTANT, REPULSION_DAMPING_FACTOR, 1.0)
]

# Main loop
last_time = glfw.get_time()
frame_count = 0
last_fps_time = last_time

while not glfw.window_should_close(window):
    current_time = glfw.get_time()
    dt = current_time - last_time
    last_time = current_time

    # Physics update
    for field in potential_fields:
        field.apply_force(entities)
    for entity in entities:
        entity.update(dt)

    # Update VBO data
    data = np.array([(e.x, e.y, e.radius, *e.color) for e in entities], dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)

    # Render
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_POINTS, 0, len(entities))
    glfw.swap_buffers(window)

    # Event processing
    glfw.poll_events()

    # FPS calculation
    frame_count += 1
    if current_time - last_fps_time >= 1.0:
        fps = frame_count / (current_time - last_fps_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        last_fps_time = current_time

glDeleteBuffers(1, [vbo])
glDeleteVertexArrays(1, [vao])
glDeleteProgram(program)
glfw.terminate()