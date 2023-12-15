import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order(f, x0, t0, t1, h):
    """
    Solve the ODE x' = f(x) using 4th order Runge-Kutta method.
    :param f: Function in the differential equation x' = f(x)
    :param x0: Initial condition
    :param t0: Initial time
    :param t1: Final time
    :param h: Step size
    :return: t-values, x-values
    """
    t_values = np.arange(t0, t1, h)
    x_values = []
    x = x0

    for t in t_values:
        x_values.append(x)
        k1 = h * f(x)
        k2 = h * f(x + 0.5 * k1)
        k3 = h * f(x + 0.5 * k2)
        k4 = h * f(x + k3)
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return t_values, x_values

def polynomial(x):
    # Change this polynomial as per your requirement
    return x*x*x

x0 = 1  # Initial condition
t0 = 0  # Start time
t1 = 4  # End time
h = 0.01  # Step size

t_values, x_values = runge_kutta_4th_order(polynomial, x0, t0, t1, h)

# Plotting
plt.plot(t_values, x_values, label="x(t)")
plt.xlabel("Time (t)")
plt.ylabel("x(t)")
plt.title("Numerical Solution of x' = p(x)")
plt.legend()

# Set aspect ratio to be equal for a square plot
#plt.gca().set_aspect('equal', adjustable='box')

# Adjust x and y axis limits if necessary
# Uncomment and modify the lines below to set specific limits
plt.xlim([t0, t1])
#plt.ylim([min_value, max_value])

plt.grid(True)
plt.show()
