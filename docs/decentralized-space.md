### Distributed Networking with Toroidal Topology

#### Overview
To manage large-scale simulations, we distribute the computational load across multiple servers. Each server manages a distinct region of the toroidal space, and these regions communicate to ensure seamless simulation across boundaries.

#### Structure of the Network
We organize the servers in an \(M \times N \times O\) grid reflecting the toroidal structure. For illustration, consider a 3x3x2 grid:

**Bottom Layer:**

```
   |        |        |
- 01 -- 02 -- 03 -
   |        |        |
- 04 -- 05 -- 06 -
   |        |        |
- 07 -- 08 -- 09 -
   |        |        |
```

**Top Layer:**

```
   |        |        |
- 10 -- 11 -- 12 -
   |        |        |
- 13 -- 14 -- 15 -
   |        |        |
- 16 -- 17 -- 18 -
   |        |        |
```

Each server is connected to its neighbors, considering wrap-around effects. For example, server 02 is connected to:
- 03 (x-right)
- 01 (x-left)
- 05 (y-down)
- 08 (y-up)
- 11 (z-up)
- 11 (z-down)

#### Data Exchange Between Servers
Servers exchange data for particles near their boundaries, ensuring proper handling of particles crossing from one server's region to another. This involves:
1. **Boundary Detection**: Detect particles near boundaries.
2. **Position Adjustment**: Adjust positions for wrap-around.
3. **Data Transfer**: Transfer particle data to the adjacent server.

For potential fields, servers treat non-adjacent nodes as single point masses/charges to simplify calculations, focusing detailed interactions on adjacent servers.

#### Visualization
Below is a diagram illustrating the connectivity in a 3x3x2 grid of servers with toroidal wrap-around:

**Bottom Layer:**

```
   |        |        |
- 01 -- 02 -- 03 -
   |        |        |
- 04 -- 05 -- 06 -
   |        |        |
- 07 -- 08 -- 09 -
   |        |        |
```

**Top Layer:**

```
   |        |        |
- 10 -- 11 -- 12 -
   |        |        |
- 13 -- 14 -- 15 -
   |        |        |
- 16 -- 17 -- 18 -
   |        |        |
```

Arrows indicate the wrap-around connections. For example, moving left from server 03 wraps around to the right side of server 01, and moving up from server 02 wraps around to the bottom of server 05.

