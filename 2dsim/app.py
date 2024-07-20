from fastapi import FastAPI
from fastapi.responses import RedirectResponse  # Add this import
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ctypes
import mmap
import os
import time

app = FastAPI()

# Mount the static directory to serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

N = 3  # Number of bodies

# Define ctypes structure
class CBody(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("velocity", ctypes.c_float * 3),
        ("mass", ctypes.c_float),
    ]

# Define Pydantic model
class Body(BaseModel):
    position: list[float]
    velocity: list[float]
    mass: float

# Retry logic to open shared memory segment
while True:
    try:
        shm_fd = os.open("/dev/shm/bodies", os.O_RDWR)
        print("Shared memory segment /dev/shm/bodies opened successfully.")
        break
    except FileNotFoundError:
        print("Shared memory segment /dev/shm/bodies not found. Retrying in 1 second...")
        time.sleep(1)

try:
    shm = mmap.mmap(shm_fd, N * ctypes.sizeof(CBody))
    print("Memory-mapped successfully.")
except Exception as e:
    print(f"Error during mmap: {e}")
    exit(1)

# Create ctypes array from shared memory
shared_bodies = (CBody * N).from_buffer(shm)

@app.get("/bodies")
async def get_bodies():
    bodies = [Body(
        position=list(shared_bodies[i].position),
        velocity=list(shared_bodies[i].velocity),
        mass=shared_bodies[i].mass
    ) for i in range(N)]
    return bodies

@app.get("/")
async def read_index():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
