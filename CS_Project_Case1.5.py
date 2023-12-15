import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the nonlinear differential equation
def nonlinear_diff_eq(t, x):
    return (x - 1) * (x - 2) * x

# Define the transformation
def transformation(x):
    if x <= 0 or x >= 2:  # Adjusted to avoid negative values under the sqrt
        return np.nan
    return np.sqrt(x * (x - 2)) / (x - 1)

# Define the linearized differential equation
def linearized_diff_eq(t, y):
    return y

# Initial conditions and time span
x0 = 0.5  # Initial condition for x
y0 = transformation(x0)  # Initial condition for y
t_span = [0, 2]  # Start and end times
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points for evaluation

# Solve the nonlinear equation numerically using solve_ivp
sol_x = solve_ivp(nonlinear_diff_eq, t_span, [x0], t_eval=t_eval, max_step=0.1)

# Solve the linearized equation numerically using solve_ivp
sol_y = solve_ivp(linearized_diff_eq, t_span, [y0], t_eval=t_eval, max_step=0.1)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(sol_x.t, sol_x.y[0], label='Numerical solution of x\'', linewidth=2)
plt.plot(sol_y.t, sol_y.y[0], '--', label='Solution of y\'', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Comparison of Solutions for x\' and y\'', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
