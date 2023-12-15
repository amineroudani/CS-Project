import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy.integrate import solve_ivp

# Define the nonlinear differential equation as a function
def nonlinear_diff_eq(t, x):
    return (x - 1) * (x - 2) * x

# Define the time span and the initial conditio

# Define the transformation
def transformation(x):
    # Avoid division by zero or very small values
    if np.isclose(x, 1):
        return np.inf
    else:
        return np.sqrt(x * (x - 2)) / (x - 1)
# Define the inverse transformation
def inverse_transform(y):
    # Ensure y is within the valid range to prevent complex solutions
    # Note: This inverse transformation formula may need validation for correctness
    #return (2 * y**2 - 2 + np.sqrt((2 * y**2 + 2)**2 - 4 * (y**2 - 1) * y**2)) / (2 * (y**2 - 1))
    if y <= 1:
        return np.nan  # Return NaN for values outside the domain
    return (y**2 - 1 - np.sqrt(y**2 -1 )) / (y**2 - 1)



t_span = (0, 5)
x0 = [2.5]  # Initial condition for x (as a list for solve_ivp)
y0 = transformation(x0[0])  # Transform initial condition for y



# Solve the nonlinear equation numerically using solve_ivp
solution = solve_ivp(nonlinear_diff_eq, t_span, x0, t_eval=np.linspace(t_span[0], t_span[1], 100),
                     method='BDF',  # This method is suited for stiff problems
                     atol=1e-6,  # Set a smaller absolute tolerance
                     rtol=1e-6)  # Set a smaller relative tolerance

# Extract the solution
t = solution.t
x_numerical = solution.y[0]

# Solve the linear equation numerically
y_linear = odeint(lambda y, t: y, y0, t).flatten()

# Apply the inverse transformation to get x from y
x_from_y = [inverse_transform(yi) for yi in y_linear]  # Using 
# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, x_numerical, label="Numerical solution of x'", linewidth=2)
plt.plot(t, x_from_y, '--', label="Solution from inverse-transformed y", linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Comparison of Numerical Solution and Inverse-Transformed Solution', fontsize=16)
#plt.ylim(-1, 2)
plt.legend()


plt.grid(True)

plt.show()



data = np.column_stack((t, x.flatten(), x_from_y.flatten()))
df = pd.DataFrame(data, columns=["time", "x", "x_from_y"])
df.to_csv("data3.csv", index=False)