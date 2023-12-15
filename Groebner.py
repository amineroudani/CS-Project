from sympy import symbols, groebner
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# Define the variables
# y1 = e^t
# y2 = e^2t
# x = y1 + y2
# x = e^t + e^2t
# ×' = e^t + 2 * e^2t
# e^t = z
# × = z + z^2
# ×' = z + 2z^2


z, x, xp = symbols('z x xp')

# Define the equations
eqs = [x - (z + z**2), xp - (z + 2 * z**2)]

# Compute the Groebner basis
g_basis = groebner(eqs, z, x, xp)

# Print the Groebner basis
print(g_basis)


# Define the analytical solution
def analytical_solution(t):
    return np.exp(t) + np.exp(2*t)

# Define the numerical root-finding for the algebraic equation
def find_x(xp_val, x_guess):
    # Define the algebraic equation from the Groebner basis
    def equation(x):
        return 4*x**2 - 4*x*xp_val + x + xp_val**2 - xp_val
    # Solve the equation using a numerical root finder
    x_solution, = fsolve(equation, x_guess)
    return x_solution

# Define the ODE to solve numerically
def ode_system(xp, t):
    # x_guess is the last computed value of x, initially set to the analytical solution at t=0
    x_guess = analytical_solution(t)
    x_val = find_x(xp, x_guess)
    return x_val + 2*x_val**2

# Initial condition for xp (x')
xp0 = 1 + 2  # Because x' = e^t + 2e^2t, and at t=0, e^0 = 1

# Time points to solve over
t = np.linspace(0, 1, 100)

# Solve the differential equation numerically
xp_numerical = odeint(ode_system, xp0, t).flatten()

# Compute x from xp numerically
x_numerical = np.array([find_x(xp_val, analytical_solution(ti)) for ti, xp_val in zip(t, xp_numerical)])

# Compute the analytical solution
x_analytical = analytical_solution(t)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, x_analytical, label="Analytical Solution", linewidth=2)
plt.plot(t, x_numerical, 'o', label="Numerical Solution", markersize=3)
plt.xlabel('Time', fontsize=14)
plt.ylabel('x', fontsize=14)
plt.title('Comparison of Analytical and Numerical Solutions', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
