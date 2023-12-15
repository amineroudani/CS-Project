import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd


#Define scale factor to make graph readable
scale_factor = 1
# Define the nonlinear differential equation
def nonlinear_diff_eq(x, t):
    """Nonlinear differential equation: x' = (x - 1)(x - 2)x"""
    return  (x - 1) * (x - 2) * x

# Define the transformation
def transformation(x):
    # Avoid division by zero or very small values
    if np.isclose(x, 1):
        return np.inf
    else:
        return np.sqrt(x * (x - 2) / (x - 1)**2)
# Define the inverse transformation
def inverse_transform(y):
    # Ensure y is within the valid range to prevent complex solutions
    # Note: This inverse transformation formula may need validation for correctness
    #return (2 * y**2 - 2 + np.sqrt((2 * y**2 + 2)**2 - 4 * (y**2 - 1) * y**2)) / (2 * (y**2 - 1))
    if y <= 1:
        return np.nan  # Return NaN for values outside the domain
    return (y**2 - 1 - np.sqrt(y**2 -1 )) / (y**2 - 1)



# Initial conditions
x0 = 4  # Initial condition for x must be bigger than 2...
y0 = transformation(x0)  # Transform initial condition for y
print(y0)
t = np.linspace(0, 5, 100)  # Time points

# Solve the nonlinear equation numerically
#x_numerical = odeint(nonlinear_diff_eq, x0, t).flatten()

# Solve the linear equation numerically
y_linear = odeint(lambda y, t: y, y0, t).flatten()


plt.figure(figsize=(10, 6))
plt.plot(t, x_from_y, label="Numerical solution of x'", linewidth=2)
plt.plot(t, x_from_y, '--', label="Solution from inverse-transformed y", linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Comparison of Numerical Solution and Inverse-Transformed Solution', fontsize=16)
plt.ylim(-1, 2)
plt.legend()


plt.grid(True)

plt.show()

