import numpy as np
import matplotlib.pyplot as plt

# Define the CDF function
def cdf(t, rate):
    return 1 - np.exp(-rate * t)

# Generate data points
t = np.linspace(0, 2, 400)
cdf_energy_1 = cdf(t, 1)
cdf_energy_4 = cdf(t, 4)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(t, cdf_energy_1, label='Energy 1')
plt.plot(t, cdf_energy_4, label='Energy 4')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('CDF of Exponential Distribution')
plt.grid(True)
plt.legend()

# Remove x-axis tick marks
plt.gca().xaxis.set_ticks([])

# Show the plot
plt.show()
