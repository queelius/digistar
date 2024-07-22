import asyncio
import websockets
import json
import mmap
import struct

N = 3  # Number of bodies
SIZE_OF_BODY = 56  # Size of each Body struct in bytes (7 doubles)

async def send_data(websocket, path):
    try:
        with open("/dev/shm/bodies", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            while True:
                bodies = []
                for i in range(N):
                    data = mm.read(SIZE_OF_BODY)
                    if len(data) < SIZE_OF_BODY:
                        mm.seek(0)  # Reset file pointer to beginning
                        continue  # Skip incomplete data
                    position = struct.unpack('3d', data[:24])
                    velocity = struct.unpack('3d', data[24:48])
                    mass = struct.unpack('d', data[48:56])[0]
                    body = {
                        'position': {'x': position[0], 'y': position[1], 'z': position[2]},
                        'velocity': {'x': velocity[0], 'y': velocity[1], 'z': velocity[2]},
                        'mass': mass
                    }
                    bodies.append(body)
                await websocket.send(json.dumps(bodies))
                await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    async with websockets.serve(send_data, "localhost", 8080):
        await asyncio.Future()  # Run forever

asyncio.run(main())
