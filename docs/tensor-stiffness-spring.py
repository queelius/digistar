import numpy as np
import matplotlib.pyplot as plt

# Define stiffness tensor
kappa = np.array([[0.1      , .1    , -.5],
                  [.1       , 1     , .8],
                  [-.5      , .8    , 10]])

# Detect that it satisfies the necessary conditions for a stiffness tensor
assert np.allclose(kappa, kappa.T), "Stiffness tensor must be symmetric"
assert np.all(np.linalg.eigvals(kappa) > 0), "Stiffness tensor must be positive definite"

# Simulation parameters
delta_t = 0.01  # time step
num_steps = 500000  # number of steps

# Initial conditions
X_red = np.array([1.2, -0.5, 0], dtype=np.float64)
X_green = np.array([0, 2.5, 0], dtype=np.float64)
V_red = np.array([0, 0, 0], dtype=np.float64)
V_green = np.array([0, 0, 0], dtype=np.float64)
m_red = 1
m_green = 1e20


# set equilibrium distance to be starting distance plus small perturbation eps
eps = .01
L = np.linalg.norm(X_red - X_green) + eps

# Store positions for visualization
positions_red = np.zeros((num_steps, 3), dtype=np.float64)
positions_green = np.zeros((num_steps, 3), dtype=np.float64)
positions_red[0] = X_red
positions_green[0] = X_green

# Simulation loop
for step in range(1, num_steps):
    # Calculate distance
    d = np.linalg.norm(X_red - X_green)

    # Let's compute equilibrium distance. It should be in the same direction as
    # the vector d = X_red - X_green but normalized to length L. we define it dL
    d = X_red - X_green
    dL = L * d / np.linalg.norm(d)
    F_red = -kappa @ (d - dL)
    F_green = -F_red
    
    # Update velocities
    V_red += F_red / m_red * delta_t
    V_green += F_green / m_green * delta_t
    
    # Update positions
    X_red += V_red * delta_t
    X_green += V_green * delta_t
    
    # Store positions
    positions_red[step] = X_red
    positions_green[step] = X_green

# Plotting the dynamics
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(positions_red[:, 0], positions_red[:, 1], positions_red[:, 2], label='Red Trajectory')
#ax.plot(positions_green[:, 0], positions_green[:, 1], positions_green[:, 2], label='Green Trajectory')
ax.scatter(positions_red[0, 0], positions_red[0, 1], positions_red[0, 2], color='red', label='Red (start)')
ax.scatter(positions_red[-1, 0], positions_red[-1, 1], positions_red[-1, 2], color='darkred', label='Red (end)')
ax.scatter(positions_green[0, 0], positions_green[0, 1], positions_green[0, 2], color='green', label='Green (stationary)')
#ax.scatter(positions_green[-1, 0], positions_green[-1, 1], positions_green[-1, 2], color='darkgreen', label='End Green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Big Atom Dynamics: Tensor-Stiffness Spring')
ax.legend()

plt.show()
