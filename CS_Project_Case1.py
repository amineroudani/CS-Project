import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd



# 1. Define the differential equation
def nonlinear_diff_eq(x, t, n):
    """Nonlinear differential equation: x' = x^n"""
    return x**n

def linearized_diff_eq(y, t, n):
    """Linearized differential equation using transformation: y' = (1-n)"""
    return (1 - n)

# Choose a random n between 2 and 4 for demonstration
n = 5
print(f"Chosen n: {n}")

x0 = 0.5  # initial condition
#t = np.linspace(0, 2, 100)  # time points
t = np.linspace(0, 1.85, 50)


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







data = np.column_stack((t, x.flatten(), x_from_y.flatten()))
df = pd.DataFrame(data, columns=["time", "x", "x_from_y"])
df.to_csv("data.csv", index=False)



# # Time points
# t = np.linspace(0, 1.85, 50)

# # Initial condition
# x0 = 0.5

# # Create a DataFrame to store all the data
# columns = ['time']
# data = {'time': t}

# # Solve each differential equation and its linearized form
# for n in range(2, 6):
#     x = odeint(nonlinear_diff_eq, x0, t, args=(n,))
#     y = odeint(linearized_diff_eq, x0**(1-n), t, args=(n,))
#     x_from_y = y**(1 / (1 - n))

#     data[f'x_{n}'] = x.flatten()
#     data[f'x_from_y_{n}'] = x_from_y.flatten()
#     columns.extend([f'x_{n}', f'x_from_y_{n}'])

# df = pd.DataFrame(data, columns=columns)
# df.to_csv("combined_data.csv", index=False)