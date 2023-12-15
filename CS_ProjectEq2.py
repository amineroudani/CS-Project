import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. Define the differential equations
def nonlinear_diff_eq2(x, t, a, b):
    """Nonlinear differential equation: x' = (x-a)(x-b)"""
    return (x - a) * (x - b)

def linearized_diff_eq2(y, t, a, b):
    """Linearized differential equation using transformation: y' = -1 + (a-b)y"""
    return -1 + (a - b) * y

# Choose random values for a and b for demonstration
a = 3
b = 2
print(f"Chosen a: {a:.2f}, Chosen b: {b:.2f}")

x0 = 1  # initial condition
t = np.linspace(0, 2, 100)  # time points

# 2. Solve nonlinear equation numerically
x = odeint(nonlinear_diff_eq2, x0, t, args=(a, b))

# 3. Apply the transformation and solve the linearized equation
y0 = 1 / (x0 - a)  # transformed initial condition
y = odeint(linearized_diff_eq2, y0, t, args=(a, b))

# Transform y back to x for comparison using the inverse transformation
x_from_y = a + 1/y

# 4. Plot both solutions
plt.figure(figsize=(10,6))
plt.plot(t, x, label=f"Numerical solution of x' = (x-{a:.2f})(x-{b:.2f})", linewidth=2)
plt.plot(t, x_from_y, '--', label='Solution from transformation', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('x(t)', fontsize=14)
plt.ylim(0, 5)  # Adjust the y-axis to prevent blowup from affecting visibility
plt.title(f'Comparison of DE Solutions', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
