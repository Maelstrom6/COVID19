"""
This class contains all parameters for all models for different countries.
It contains methods to obtain observed data.
It also contains the common methods to use the model itself.
"""

import numpy as np
import pandas as pd
import math
from Communication import Database

np.set_printoptions(suppress=True)


# Inspiration for the model:
# https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3182455/
# https://www.researchgate.net/publication/334695153_Complete_maximum_likelihood_estimation_for_SEIR_epidemic_models_theoretical_development
# I got up to the first sample path diagrams before I got completely lost

class Parameters:
    """
    A single class to store all model parameters.
    See model_1 comments for information on these parameters
    """

    def __init__(self, m, n, Ep, Ip, alpha, beta, offset, country, is_country, N):
        self.m = m
        self.n = n
        self.Ep = Ep
        self.Ip = Ip
        self.alpha = alpha
        self.beta = beta
        self.offset = offset
        self.country = country
        self.is_country = is_country
        self.N = N

    def unpack(self):
        """
        :return: all the model parameters as a tuple
        """
        return self.m, self.n, self.Ep, self.Ip, self.alpha, self.beta, self.offset, \
               self.country, self.is_country, self.N


def choose(n, r):
    return (math.factorial(n)) / math.factorial(r) / math.factorial(n - r)


def binomial(x, n, p):
    return choose(n, x) * p ** x * (1 - p) ** (n - x)


def binomial_dist(n, p):
    dist = []
    for x in range(n + 1):
        dist.append(binomial(x, n, p))
    return np.array(dist)


def neg_bin(x, k, p):
    return choose(x + k - 1, k - 1) * p ** k * (1 - p) ** x


def neg_bin_dist(k, p, length):
    dist = []
    for x in range(length):
        dist.append(neg_bin(x, k, p))
    return np.array(dist)


def get_observed_I_and_R(country: str, is_country: bool = True):
    """
    Gets the data for the number of confirmed cases and the number of recovered.

    :param country: The country or province for which to get the cases
    :param is_country: Whether the country variable is for a country or for a province
    :return: A tuple of 2 arrays: one for cumulative infected and one for cumulative recovered
    """
    country = country.replace("'", "")
    database = Database("Data\\CombinedData.accdb")
    if is_country:
        # Source: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
        # And https://www.worldometers.info/coronavirus/
        data_dict = database.select("SELECT IRTbl.Country, IRTbl.Day, Sum(IRTbl.Infected) AS SumOfInfected, "
                                    "Sum([IRTbl].[Recovered]+[IRTbl].[Dead]) AS SumOfRecoveredAndDead\n"
                                    "FROM IRTbl\n"
                                    "GROUP BY IRTbl.Country, IRTbl.Day\n"
                                    "HAVING (((IRTbl.Country)='{}') AND ((Sum(IRTbl.Infected))>0))\n"
                                    "ORDER BY IRTbl.Day;".format(country),
                                    ["SumOfInfected", "SumOfRecoveredAndDead"])
    else:
        print("Warning: Worldometer (the data source since 2020/03/25) does not specify provinces in their data.\n"
              "         It is best to rather specify a country than a province due to this.")
        data_dict = database.select("SELECT IRTbl.Province, IRTbl.Day, Sum(IRTbl.Infected) AS SumOfInfected, "
                                    "Sum([IRTbl].[Recovered]+[IRTbl].[Dead]) AS SumOfRecoveredAndDead\n"
                                    "FROM IRTbl\n"
                                    "GROUP BY IRTbl.Province, IRTbl.Day\n"
                                    "HAVING (((IRTbl.Province)='{}') AND ((Sum([IRTbl].[Infected]))>0))\n"
                                    "ORDER BY IRTbl.Day;".format(country),
                                    ["SumOfInfected", "SumOfRecoveredAndDead"])

    data = pd.DataFrame(data=data_dict)
    data.columns = data.columns.str.replace('/', '')  # Just a safety precaution
    I_tot_observed = data["SumOfInfected"].to_numpy()
    R_tot_observed = data["SumOfRecoveredAndDead"].to_numpy()

    return I_tot_observed, R_tot_observed


def model_SA():
    """
    Gets model parameters for the first type of model that seems to work.
    Here we assume a discrete SEIR model with m substates in E and n substates in I.
    We assume the number of days in state E follows Negative Binomial(m, Ep)
    And the number of days in state I follows a Negative Binomial(n, Ip)
    Where Ip and Ep are chosen so as to give the desired means.

    Assuming SA's lock down is as effective as China's.
    Assuming the reported cases for both China and SA are accurate representations
    of the number of infected people that are showing symptoms.
    Assuming no one will be infectious after the lock down.

    Note that this model can easily become a binomial recover time and exposed time.
    One needs to simply change the transition state matrix so that in each sub state, Ip and Ep are 1.
    Also note the Ip and Ep calculations need to change.

    :return: All the parameters as a tuple
    """
    m = 7  # The parameter within the distribution of E
    n = 7  # The parameter within the distribution of I

    mean_exposed_days = 6
    mean_infectious_days = 20
    # The result is that the median time in I is ~15 days while the mean time is 20
    # Source: https://ourworldindata.org/coronavirus

    Ep = m / (m + mean_exposed_days)  # The p needed to give the desired expected value
    Ip = n / (n + mean_infectious_days)  # The p needed to give the desired expected value

    # If one wants the binomial assumption
    # Ep = mean_exposed_days/m, m=20
    # Ip = mean_infectious_days/n, n=20

    offset = 5  # The offset time on the graph between theoretical and observed I
    # Source: https://en.wikipedia.org/wiki/2020_Hubei_lockdowns
    lock_down = 21 + offset

    # the average number of susceptible people exposed per day per person exposed immediately
    alpha = lambda t: 0.6 / (t + 1) ** 0.2 if t < lock_down else 0
    # the average number of susceptible people exposed per day per person infected
    beta = lambda t: 1.5 / (t + 1) if t < lock_down else 0

    # The number of days since patient 0 until the country goes into lock down

    country = "South Africa"
    is_country = True
    N = 58_000_000  # The total number of people in the country
    # NOTE: other countries such as China need a province specified instead
    # and you mush change the function get_observed_I_and_R to match.

    return Parameters(m, n, Ep, Ip, alpha, beta, offset, country, is_country, N)


def model_Hubei():
    """
    This is the model for Hubei that I tried to match to South Africa.
    Since the data does not contain information on patient 0 and the
    epidemic is underway already, the initial state vector must have people
    already in the pipeline.

    :return: All the parameters as a tuple
    """
    m = 7  # The parameter within the distribution of E
    n = 7  # The parameter within the distribution of I

    mean_exposed_days = 6
    mean_infectious_days = 20

    Ep = m / (m + mean_exposed_days)  # The p needed to give the desired expected value
    Ip = n / (n + mean_infectious_days)  # The p needed to give the desired expected value

    offset = 5  # The offset time on the graph between theoretical and observed I
    lock_down = 5 + offset
    # Number of days since patient 0 until the province goes into lock down

    # the average number of susceptible people exposed per day per person exposed immediately
    a, b, c, d, e, f = [102.94127581, 59.02443664, 0.21857432, 11.55142024, 1.06625104, 7.73675499]
    alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                      abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e))  # + abs(g)
    # the average number of susceptible people exposed per day per person infected
    beta = lambda t: 0.2 * alpha(t)

    # the average number of susceptible people exposed per day per person exposed immediately
    # alpha = lambda t: 1.61 / (0.5 * t + 1) if t < lock_down else 0
    # the average number of susceptible people exposed per day per person infected
    # beta = lambda t: 1.5 / (t + 1) if t < lock_down else 0

    # This should be 5
    province = "Hubei"
    is_country = False
    N = 58_000_000  # The total number of people in the province
    return Parameters(m, n, Ep, Ip, alpha, beta, offset, province, is_country, N)


def model_SouthKorea():
    """
    This is a model for South Korea.
    Due to how much testing they do, it is probably more accurate
    to try match observations to E instead of I.
    """
    m = 4  # The parameter within the distribution of E
    n = 7  # The parameter within the distribution of I

    mean_exposed_days = 11
    mean_infectious_days = 14

    Ep = m / (m + mean_exposed_days)  # The p needed to give the desired expected value
    Ip = n / (n + mean_infectious_days)  # The p needed to give the desired expected value

    offset = 2  # The offset time on the graph between theoretical and observed I
    lock_down = 28 + offset
    # Number of days since patient 0 until the province goes into lock down
    # the average number of susceptible people exposed per day per person exposed immediately
    a, b, c, d, e, f = [3.14391294, 7.8549395, 2.02250233, 29.93859138, 2.42269529, 10.44748158]
    alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                      abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e))  # + abs(g)
    # the average number of susceptible people exposed per day per person infected
    beta = lambda t: 0.2 * alpha(t)

    # the average number of susceptible people exposed per day per person exposed immediately
    # alpha = lambda t: 1.1 / (t + 1) if t < lock_down else 1.7 / (0.9 * (t - 28) + 1)
    # the average number of susceptible people exposed per day per person infected
    # beta = alpha

    # This should be 5
    country = "Korea, South"
    is_country = True
    N = 51_000_000  # The total number of people in the province

    print("Note for South Korea, since they do so much testing, it is best to match observed cases with Exposed.")
    return Parameters(m, n, Ep, Ip, alpha, beta, offset, country, is_country, N)


def model_Italy():
    """
    A model of Italy
    """
    m = 7  # The parameter within the distribution of E
    n = 7  # The parameter within the distribution of I

    mean_exposed_days = 6
    mean_infectious_days = 14

    Ep = m / (m + mean_exposed_days)  # The p needed to give the desired expected value
    Ip = n / (n + mean_infectious_days)  # The p needed to give the desired expected value

    offset = 0  # The offset time on the graph between theoretical and observed I
    lock_down = 0 + offset
    # Number of days since patient 0 until the province goes into lock down

    # the average number of susceptible people exposed per day per person exposed immediately
    a, b, c, d, e, f = [20.90247103, 1.45905646, 9.1360014, 38.84609153, 2.16521914, 8.27952718]
    alpha = lambda t: abs(c * b / a * (t / a) ** (b - 1) * math.exp(-(t / a) ** b)) + \
                      abs(f * e / d * (t / d) ** (e - 1) * math.exp(-(t / d) ** e))  # + abs(g)
    # the average number of susceptible people exposed per day per person infected
    beta = lambda t: 0.2 * alpha(t)
    # the average number of susceptible people exposed per day per person exposed immediately
    # alpha = lambda t: 1.1 / (t + 1) if t < lock_down else 1.7 / (0.9 * (t - 28) + 1)
    # the average number of susceptible people exposed per day per person infected
    # beta = alpha

    # This should be 5
    country = "Italy"
    is_country = True
    N = 60_000_000  # The total number of people in the province
    return Parameters(m, n, Ep, Ip, alpha, beta, offset, country, is_country, N)


def create_transition_matrix(m, n, Ep, Ip):
    """
    Sets up the initial transition matrix.

    :param m: See model_1 comments
    :param n: See model_1 comments
    :param Ep: See model_1 comments
    :param Ip: See model_1 comments
    :return: The one-step transition matrix
    """
    # This will change within the loop but most of the values will remain the same
    P = np.zeros((m + n + 2, m + n + 2))
    P[0, 0] = 1
    for row in range(1, m + 1):
        P[row, row] = 1 - Ep
        P[row, row + 1] = Ep
    P[m, m + 1:m + n + 2] = Ep * np.array(binomial_dist(n, Ip))
    for row in range(m + 1, m + n + 1):
        P[row, row] = 1 - Ip
        P[row, row + 1] = Ip
    P[m + n + 1, m + n + 1] = 1
    return P


def __create_transition_matrix_binomial(m, n, Ep, Ip):
    """
    Sets up the initial transition matrix under the binomial assumption.
    This method is not advised.

    :param m: See model_1 comments
    :param n: See model_1 comments
    :param Ep: See model_1 comments
    :param Ip: See model_1 comments
    :return: The one-step transition matrix
    """
    # This will change within the loop but most of the values will remain the same
    P = np.zeros((m + n + 2, m + n + 2))
    P[0, 0] = 1
    for row in range(1, m + 1):
        P[row, row + 1] = 1
    P[m, m + 1:m + n + 2] = Ep * np.array(binomial_dist(n, Ip))
    for row in range(m + 1, m + n + 1):
        P[row, row + 1] = 1
    P[m + n + 1, m + n + 1] = 1
    return P


def create_initial_state_vector(m, n, N, N0, NR):
    """
    Creates the initial state vector for the system.

    :param NR: Number of people who have recovered
    :param m: See model_1 comments
    :param n: See model_1 comments
    :param N: The number of people in the homogenous group
    :param N0: The number of "patient 0's"
    :return: a numpy 1 by m+n+2 matrix
    """
    pt = np.zeros((1, m + n + 2))
    pt[0, 0] = 1 - 2 * N0 / N - NR / N
    pt[0, 1:m + 1] = N0 / N / m
    pt[0, m + 1:m + n + 1] = N0 / N / n
    pt[0, m + n + 1] = NR / N
    return pt


def get_modelled_time_series(params: Parameters, N0, NR, max_T, return_R_0 = False):
    """
    Runs the model using the model parameters

    :param return_R_0: Whether or not the function should also return a numpy array of R_0
    :param max_T: The number of time steps (up to an excluding) for the model
    :param params: The model parameters
    :param N0: The number of "patient 0's" at time 0
    :param NR: The number of already recovered at time 0
    :return: a tuple of numpy arrays of S,E,I,R,I_tot,R_tot
    """
    m, n, Ep, Ip, alpha, beta, _, country, is_country, N = params.unpack()
    P = create_transition_matrix(m, n, Ep, Ip)
    pt = create_initial_state_vector(m, n, N, N0, NR)

    # Set up the time series variables
    T = np.arange(0, max_T)
    S = []  # The proportion of people in the susceptible stage at time t
    E = []  # The proportion of people in the exposed stage at time t
    I = []  # The proportion of people in the infectious stage at time t
    I_tot = []  # The total number of people infected since time 0 up until time t
    E_tot = []
    R = []  # The proportion of people in the recovered stage at time t
    R_0 = []  # the basic reproduction number
    length = 25  # the maximum value of the random variable T_E to sum up to
    # This can't be large since the factorials in the negative binomial distribution get massive

    for t in T:
        # Append the current state of the system
        S.append(np.sum(pt[0, 0]))
        E.append(np.sum(pt[0, 1:m + 1]))
        I.append(np.sum(pt[0, m + 1:m + n + 1]))
        R.append(np.sum(pt[0, m + n + 1]))
        I_tot.append(I[-1] + R[-1])
        E_tot.append(E[-1] + I_tot[-1])

        # Estimate R_0 using an arbitrary alpha and beta
        # Warning: this slows down the program drastically due to sums in multiple loops
        # This is by far the biggest time consuming process in the calculation
        if return_R_0:
            ET_E = round(m * (1 - Ep) / Ep)  # E[T_E]
            total = 0
            for t_val in range(length):  # run through all possible values of T_E and T_I
                for s in range(1, t_val + 1):  # sum from 1 up to the value of T_E inclusive
                    total += alpha(t + s) * neg_bin(t_val, m, Ep)
                    total += beta(t + ET_E + s) * neg_bin(t_val, n, Ip)
            R_0.append(total)

        # Estimate R_0 using alpha_int and beta_int as Weibull anti-derivatives or sums up to n
        # This is faster but needs alpha and beta to be defined in specific ways.
        # a, b, c, d, e, f = [30.73504421, 2.5255927, 6.15897371, 22.99668146, 1.00000064, 12.86532984]
        # alpha_int = lambda w: abs(c * math.exp(-(w / a) ** b)) + abs(f * math.exp(-(w / d) ** e))
        # beta_int = lambda w: 0.2*alpha(w)
        #
        # ET_E = round(m * (1 - Ep) / Ep)  # E[T_E]
        # T_E = np.arange(length)  # Possible values for T_E and T_I to take on in the sum
        # summand = lambda s: (alpha_int(t) - alpha_int(t + s)) * neg_bin(s, m, Ep) + \
        #                     (beta_int(t + ET_E) - beta_int(t + ET_E + s)) * neg_bin(s, n, Ip)
        # R_0.append(np.sum(np.array(list(map(summand, T_E)))))

        # Adjust the transition matrix
        P_adjusted = P.copy()
        P_adjusted[0, 0] = 1 - alpha(t) * E[-1] - beta(t) * I[-1]
        P_adjusted[0, 1:m + 2] = alpha(t) * E[-1] * np.array(binomial_dist(m, Ep)) \
                                 + beta(t) * I[-1] * np.array(binomial_dist(m, Ep))  # the conditional binomial

        # Refresh the time t state vector
        pt = np.matmul(pt, P_adjusted)

    # Turn everything into a numpy array and make them numbers instead of proportions
    S = np.array(S) * N
    E = np.array(E) * N
    I = np.array(I) * N
    R = np.array(R) * N
    I_tot = np.array(I_tot) * N
    E_tot = np.array(E_tot) * N
    R_0 = np.array(R_0)

    if return_R_0:
        return S, E, I, R, I_tot, E_tot, R_0
    else:
        return S, E, I, R, I_tot, E_tot


def get_mse(params: Parameters, N0, NR, I_tot_observed):
    """
    Calculates the Mean Square Error between the modelled data and the observed data.
    Note for South Korea this must change. All instances of I_tot must change to E_tot.

    :param params: The model parameters
    :param N0: The number of "patient 0's" at time 0
    :param NR: The number of already recovered at time 0
    :param I_tot_observed: The observed total number in the infectious state
    :return: The MSE
    """
    _, _, _, _, I_tot, E_tot = get_modelled_time_series(params, N0, NR, len(I_tot_observed) + params.offset, False)
    I_tot = I_tot[params.offset:]
    mse = np.sum((I_tot - I_tot_observed) ** 2) / len(I_tot_observed)
    # Note for South Korea, by the number of tests that they are doing,
    # it is best to calculate MSE by E_tot - I_tot_observed
    return mse
