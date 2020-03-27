"""
This is the most common class to use.
This should be the first place you come to in order to see how to use it.
"""

from Models import *
import matplotlib.pyplot as plt

params = model_SA()
# See some of the params attributes and change a few if you want to.
# The description of each parameter are in the comments of model_SA()
# params.offset = 3
I_tot_observed, R_tot_observed = get_observed_I_and_R(params.country, params.is_country)

max_T = 100
T = np.arange(0, max_T)
S, E, I, R, I_tot, E_tot = get_modelled_time_series(params, I_tot_observed[0], R_tot_observed[0], max_T)

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

# log scale plot
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
plt.xlabel("Time in days since patient 0")
plt.legend()
plt.show()
