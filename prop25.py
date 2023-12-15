import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Define the new nonlinear differential equation for use with solve_ivp
def new_nonlinear_diff_eq(t, x, a, b):
    return x**2 + a*x + b

# Define the transformation and its inverse
def transformation(a, b, x):
    return np.sqrt((2*b - a**2) / (2*b - a**2 + 2*x**2))

def inverse_transformation(y, a, b):
    return np.sqrt((y**2 * (2*b - a**2) - (2*b - a**2)) / 2)

# Solve the original equation numerically using solve_ivp with Radau method
a, b = 5, 2  # Example values for a and b
x0 = [0.1]  # Initial condition for x in list format
t_span = [0, 2]  # Time span
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points for evaluation

sol_x = solve_ivp(new_nonlinear_diff_eq, t_span, x0, args=(a, b), method='Radau', t_eval=t_eval)

# Solve the linear differential equation analytically
t_sym, a_sym, b_sym = sp.symbols('t a b')
y = sp.Function('y')(t_sym)
k = -(2*b_sym - a_sym**2) / 2

# Define the ODE
ode = sp.Eq(y.diff(t_sym, t_sym) + k*y, 0)

# Solve the ODE
sol = sp.dsolve(ode, y, ics={y.subs(t_sym, 0): 1, y.diff(t_sym).subs(t_sym, 0): 0})

# Convert the solution to a function that can be evaluated
y_sol_func = sp.lambdify((t_sym, a_sym, b_sym), sol.rhs, modules=['numpy'])

# Evaluate the solution for each time point
y_analytical = np.array([y_sol_func(ti, a, b) for ti in t_eval], dtype='float')

# Apply the inverse transformation to get x from y
x_from_y_analytical = inverse_transformation(y_analytical, a, b)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(sol_x.t, sol_x.y[0], label='Numerical solution of x\' (Radau)', linewidth=2)
plt.plot(t_eval, x_from_y_analytical, '--', label='Solution from analytical y\'', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Comparison of Numerical and Analytical Solutions for x(t)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
