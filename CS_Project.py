import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. Define the differential equation
def nonlinear_diff_eq(x, t, n):
    """Nonlinear differential equation: x' = x^n"""
    return x**n

def linearized_diff_eq(y, t, n):
    """Linearized differential equation using transformation: y' = (1-n)"""
    return (1 - n)

# Choose a random n between 2 and 4 for demonstration
n = 2
print(f"Chosen n: {n}")

x0 = 0.5  # initial condition
t = np.linspace(0, 2, 100)  # time points

# 2. Solve nonlinear equation numerically
x = odeint(nonlinear_diff_eq, x0, t, args=(n,))

# 3. Apply the transformation and solve the linearized equation
y0 = x0**(1-n)  # transformed initial condition
y = odeint(linearized_diff_eq, y0, t, args=(n,))

# Transform y back to x for comparison using the inverse transformation
x_from_y = y**(1 / (1 - n))

# 4. Plot both solutions
plt.figure(figsize=(10,6))
plt.plot(t, x, label=f'Numerical solution of x\' = x^{n:.2f}', linewidth=2)
plt.plot(t, x_from_y, '--', label='Solution from transformation', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('x(t)', fontsize=14)
plt.ylim(0, 10)  # Adjust the y-axis to prevent blowup from affecting visibility
plt.title(f'Comparison of DE Solutions', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
