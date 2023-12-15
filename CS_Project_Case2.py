import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd


# 1. Define the nonlinear differential equation
def nonlinear_diff_eq(x, t, a, b):
    """Nonlinear differential equation: x' = (x-a)(x-b)"""
    return (x - a) * (x - b)

def linearized_diff_eq(y, t, a, b):
    """Linearized differential equation using transformation: y' = -1 + (a-b)y"""
    return -1 + (b - a) * y

# Choose values for a and b
a, b = 1, 3
print(f"Chosen a: {a}, b: {b}")

x0 = 0.5  # initial condition for x
t = np.linspace(0, 2, 100)  # time points

# 2. Solve nonlinear equation numerically
x = odeint(nonlinear_diff_eq, x0, t, args=(a, b))

# 3. Apply the transformation and solve the linearized equation
y0 = 1 / (x0 - a)  # transformed initial condition for y
y = odeint(linearized_diff_eq, y0, t, args=(a, b))

# Transform y back to x for comparison using the inverse transformation
x_from_y = 1 / y + a

# 4. Plot both solutions
plt.figure(figsize=(10,6))
plt.plot(t, x, label=f'Numerical solution of x\' = (x-{a})(x-{b})', linewidth=2)
plt.plot(t, x_from_y, '--', label='Solution from transformation', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('x(t)', fontsize=14)
plt.ylim(0, 10)  # Adjust the y-axis to prevent blowup from affecting visibility
plt.title(f'Comparison of DE Solutions', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





data = np.column_stack((t, x.flatten(), x_from_y.flatten()))
df = pd.DataFrame(data, columns=["time", "x", "x_from_y"])
df.to_csv("data3.csv", index=False)