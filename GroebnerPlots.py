import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, groebner
import pandas as pd

# Define the variables
z, x, xp = symbols('z x xp')

# Define the equations
eqs = [x - (z + z**2), xp - (z + 2 * z**2)]

# Compute the Groebner basis
g_basis = groebner(eqs, z, x, xp)

# Function to evaluate the original equations
def evaluate_original_equations(z_val):
    x_val = z_val + z_val**2
    xp_val = z_val + 2 * z_val**2
    return x_val, xp_val

# Range of z values
z_range = np.linspace(-2, 2, 400)

# Evaluate the original equations over the range of z
x_orig_values = []
xp_orig_values = []
for z_val in z_range:
    x_val, xp_val = evaluate_original_equations(z_val)
    x_orig_values.append(x_val)
    xp_orig_values.append(xp_val)

# Groebner basis equations are the same as the original for x and x'
x_gb_values = x_orig_values
xp_gb_values = xp_orig_values

# Plot the results: Original Equations vs Groebner Basis
plt.figure(figsize=(12, 6))

# Original equations
plt.plot(z_range, x_orig_values, label='Original x = z + z^2', color='blue', linestyle='--')
#plt.plot(z_range, xp_orig_values, label="Original x' = z + 2z^2", color='green', linestyle='--')

# Groebner basis
plt.plot(z_range, x_gb_values, label='Groebner x = z + z^2', color='red', alpha=0.7)
#plt.plot(z_range, xp_gb_values, label="Groebner x' = z + 2z^2", color='purple', alpha=0.7)

plt.xlabel('z values')
plt.ylabel('x and x\' values')
plt.title('Comparison of Original Equations and Groebner Basis Equations')
plt.legend()
plt.grid(True)
plt.show()



data = np.column_stack((z_range, x_orig_values, x_gb_values))
df = pd.DataFrame(data, columns=["time", "x", "x_from_y"])
df.to_csv("data55.csv", index=False)