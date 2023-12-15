import numpy as np
import matplotlib.pyplot as plt

def dy_dt(y):
    """Differential equation: y' = -1/y^2."""
    return -1/y**2

def euler_method(y0, t0, tf, dt):
    """Euler's method to solve the differential equation."""
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps + 1)
    y = np.zeros(num_steps + 1)
    y[0] = y0
    
    for i in range(num_steps):
        y[i + 1] = y[i] + dy_dt(y[i]) * dt

    return t, y

# Parameters
y0 = 1        # initial condition
t0 = 0        # start time
tf = 2        # end time
dt = 0.01     # time step

# Solve using Euler's method
t, y = euler_method(y0, t0, tf, dt)

# Plot the solution
plt.plot(t, y, label="Euler's Method")
plt.xlabel('t')
plt.ylabel('y')
plt.title("Solution to y' = -1/y^2 using Euler's Method")
plt.legend()
plt.grid(True)
plt.show()
