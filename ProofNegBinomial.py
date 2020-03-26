"""
This serves as a 'proof' or demonstration that the chosen format of the transition matrix P
does in fact produce Negative binomial times spent in super-states I (and E can be shown similarly).
"""

from Models import *
import matplotlib.pyplot as plt


def neg_bin(x, k, p):
    return choose(x + k - 1, k - 1) * p ** k * (1 - p) ** x


def neg_bin_dist(k, p, length):
    dist = []
    for x in range(length):
        dist.append(neg_bin(x, k, p))
    return dist


# Some parameters that can be changed freely
m = 3
n = 7
mean_exposed_days = 14
mean_infectious_days = 20

# The resulting Ep and Ip
Ep = m / (m + mean_exposed_days)  # The p needed to give the desired expected value
Ip = n / (n + mean_infectious_days)  # The p needed to give the desired expected value

# Set up the initial state matrix to look like an individual has just left state E
pt = np.zeros((1, m + n + 2))
pt[0, m + 1:m + n + 2] = np.array(binomial_dist(n, Ip))
print(pt)

# Set up the transition matrix
# make the last line 0 so we count the number of newly recovered instead of the total recovered
P = create_transition_matrix(m, n, Ep, Ip)
P[-1, :] = np.zeros((1, m+n+2))
print(P)

# Set up the time series arrays
T = np.arange(0, 50)
R = []

for t in T:
    R.append(pt[0, -1])
    pt = np.matmul(pt, P)

print(np.sum(T * R))  # Mean of the markov chain up to the maximum value of T
plt.scatter(T, R, label="Markov chain", s=20)

# Now to compare to the actual negative binomial distribution
Iq = 1-Ip
print(n * Iq / Ip)  # The theoretical mean of a binomial distribution

plt.scatter(T, neg_bin_dist(n, Ip, len(T)), label="Neg binomial", s=4)
plt.legend()
plt.show()
