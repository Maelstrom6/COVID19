"""
This file is used to estimate alpha and beta in various models based on observed data.
"""

from Models import *
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

params = model_SA()
I_tot_observed, R_tot_observed = get_observed_I_and_R(params.country, params.is_country)


# Define the function that we want to minimise
# Note numpy will point out some warnings while minimising f(x)
def f(x):
    a, b, c, d, e, f, g = x
    params.alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                             abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e)) + abs(g)
    params.beta = lambda t: 0.2 * params.alpha(t)
    return get_mse(params, I_tot_observed[0], R_tot_observed[0], I_tot_observed)


# Set initial guess and minimise f(x)
# If you want to lower the tolerance for a higher accuracy, you can, but with so many parameters,
# I'm not sure if it will make much of a difference to the quality of prediction
x0 = [20, 3, 10, 20, 1.1, 10, 0.02]
x_opt = minimize(f, x0, method='Nelder-Mead', tol=0.01).x
print(x_opt)

# Get optimised parameters and apply them
a, b, c, d, e, f, g = x_opt
params.alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                         abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e)) + abs(g)
params.beta = lambda t: 0.2 * params.alpha(t)

max_T = 100
T = np.arange(0, max_T)
S, E, I, R, I_tot, E_tot, R_0 = get_modelled_time_series(params, I_tot_observed[0], R_tot_observed[0], max_T, True)

# Plot everything
plt.plot(T - params.offset, E_tot, color="orange", label="Cumulative Exposed")
plt.plot(T - params.offset, I_tot, color="red", label="Cumulative Infected")
plt.plot(T - params.offset, R, color="green", label="Cumulative Recovered")
plt.scatter((T[:len(I_tot_observed)]), I_tot_observed, color="red", label="Observed Infected", s=1)
plt.scatter((T[:len(R_tot_observed)]), R_tot_observed, color="green", label="Observed Recovered", s=1)
plt.title("Population dynamics over time")
plt.ylabel("Number of people")
plt.xlabel("Time in days since patient 0")
plt.legend()
# plt.savefig("SE"+str(m)+"I"+str(n)+"R.png")
plt.show()

# log scale
plt.plot(T - params.offset, np.log(S + 1), color="blue", label="Susceptible")
plt.plot(T - params.offset, np.log(E_tot + 1), color="orange", label="Exposed")
plt.plot(T - params.offset, np.log(I_tot + 1), color="red", label="Cumulative Infected")
# plt.plot(T - offset, np.log( I + 1), color="pink", label="Infected")
plt.plot(T - params.offset, np.log(R + 1), color="green", label="Recovered")
plt.scatter(np.array(T[:len(I_tot_observed)]), np.log(I_tot_observed + 1), color="red",
            label="Observed Infected", s=1)
plt.scatter(np.array(T[:len(R_tot_observed)]), np.log(R_tot_observed + 1), color="green",
            label="Observed Recovered", s=1)
plt.title("Population dynamics over time")
plt.ylabel("Log number of people")
plt.xlabel("Time in days since patient 0")
plt.legend()
# plt.savefig("SE"+str(m)+"I"+str(n)+"R log.png")
plt.show()

plt.plot(T - params.offset, list(map(params.alpha, T)), color="orange", label="Alpha")
plt.plot(T - params.offset, list(map(params.beta, T)), color="red", label="Beta")
plt.plot(T - params.offset, R_0, color="blue", label="R_0")
plt.xlabel("Time in days since patient 0")
plt.legend()
plt.show()
