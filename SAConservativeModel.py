"""
This file presents a conservative estimate of the infections in SA.
The biggest reason this is conservative is due to the Weibull assumptions
meaning that alpha and beta drop to 0 eventually.
Other shortcomings are expressed in the README.md that will also mostly result in more conservative outputs.
"""

from Models import *
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

params = model_SA()
I_tot_observed, R_tot_observed = get_observed_I_and_R(params.country, params.is_country)

# Apply optimised parameters
a, b, c, d, e, f = [30.73504421, 2.5255927,  6.15897371, 22.99668146, 1.00000064, 12.86532984]
# Old using data from 2 days ago shows a much better picture:
# a, b, c, d, e, f = [32.76717857, 1.22759355, 7.65017755, 21.02088014, 1.21173884, 9.71998687]
params.alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                         abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e))
params.beta = lambda t: 0.2 * params.alpha(t)

max_T = 100
T = np.arange(0, max_T)
S, E, I, R, I_tot, E_tot = get_modelled_time_series(params, I_tot_observed[0], R_tot_observed[0], max_T)

# Set up a date version of T. Instead of integer indices we have date indices
base = datetime.datetime(2020, 3, 5)
dates = np.array([base + datetime.timedelta(days=i) for i in range(0, max_T)])

# Plot everything
plt.plot(dates - datetime.timedelta(days=params.offset), E_tot, color="orange", label="Cumulative Exposed")
plt.plot(dates - datetime.timedelta(days=params.offset), I_tot, color="red", label="Cumulative Infected")
plt.plot(dates - datetime.timedelta(days=params.offset), R, color="green", label="Cumulative Recovered")
plt.scatter(dates[:len(I_tot_observed)], I_tot_observed, color="red", label="Observed Infected", s=5)
plt.scatter(dates[:len(R_tot_observed)], R_tot_observed, color="green", label="Observed Recovered", s=5)
plt.title("Population dynamics over time")
plt.ylabel("Number of people in SA")
plt.xlabel("Date")
plt.legend()
plt.gcf().autofmt_xdate()
# plt.savefig("SE"+str(m)+"I"+str(n)+"R.png")
plt.show()

# log scale plot
plt.plot(dates - datetime.timedelta(days=params.offset), np.log(S + 1), color="blue", label="Susceptible")
plt.plot(dates - datetime.timedelta(days=params.offset), np.log(E_tot + 1), color="orange", label="Cumulative Exposed")
plt.plot(dates - datetime.timedelta(days=params.offset), np.log(I_tot + 1), color="red", label="Cumulative Infected")
# plt.plot(T - offset, np.log( I + 1), color="pink", label="Infected")
plt.plot(dates - datetime.timedelta(days=params.offset), np.log(R + 1), color="green", label="Cumulative Recovered")
plt.scatter(dates[:len(I_tot_observed)], np.log(I_tot_observed + 1), color="red", label="Observed Infected", s=5)
plt.scatter(dates[:len(I_tot_observed)], np.log(R_tot_observed + 1), color="green", label="Observed Recovered", s=5)
plt.title("Log population dynamics over time")
plt.ylabel("Log number of people in SA")
plt.xlabel("Date")
plt.legend()
plt.gcf().autofmt_xdate()
# plt.savefig("SE"+str(m)+"I"+str(n)+"R log.png")
plt.show()

plt.plot(dates - datetime.timedelta(days=params.offset), list(map(params.alpha, T)), color="orange", label="Alpha")
plt.plot(dates - datetime.timedelta(days=params.offset), list(map(params.beta, T)), color="red", label="Beta")
plt.title("Parameter dynamics over time")
plt.ylabel("Value")
plt.xlabel("Date")
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()
