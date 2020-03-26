# COVID-19 Predictions

Some multi-stage discrete SEIR modelling of the COVID-19 disease.

# The model

This project tries to model the COVID population spread using an n-step discrete Markov SEIR model.
I assume from here on you will have read some the [Wikipedia page](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) or watched [Numberphile's video](https://www.youtube.com/watch?v=k6nLfCbAzgo).
Consider there are 4 compartments: S for susceptible, E for exposed, I for infected and R for recovered.

- S is the proportion (or number) of people who have not contracted the disease but are still able to catch it.
- E is the proportion (or number) of people who have contracted the disease and can infect others but are not showing any symptoms.
- I is the proportion (or number) of people who have contracted the disease and can infect others and are showing symptoms.
- R is the proportion (or number) of people who have recovered from the disease or have passed away from it.

Almost every one in the population follows the following stages: S -> E -> I -> R.


Now, at every time, say t, the whole state of the process can be described by the 4 numbers: S, E, I and R.
It is only natural to combine them into a row vector and we will call this 'the state vector at time t': 
```p(t) = (S, E, I, R)```.
We usually have ```p0 = (1 - N0/N, N0/N, 0, 0)``` where N is the total population in this homogeneous group and N0 is 
the number of 'patient 0s'.


Also, we need to describe the transition rates between these states.
Say an individual is susceptible at time t. Then the probability that they become exposed at time t+1 could be the 
proportion of exposed times some factor alpha plus the proportion of infected times some factor beta.
That is,  ```Pr(S -> E) = E*α + I*β```.
Both alpha and beta can be functions of t. The probability that they immediately become infected we will say is 0 and 
for becoming recovered we set to 0 too. The probability of remaining susceptible is the complement of becoming exposed. 
We repeat this process for someone who is exposed, infected and recovered and we obtain 16 different probabilities 
of transitioning between stages. Note that with this, one must also have a probability for transition from exposed 
to infected, ```Ep``` and a probability for a transition from infected to recovered, ```Ip```. 

It is also natural to combine these probabilities into a matrix and we will call this matrix the transition matrix: 
```P(t) = {{Pr(i -> j)}}i,j```.
This transition matrix is also usually a function of time t since alpha and beta are usually functions of t. We have 
something like the following:
```
       [1-E*α+I*β  E*α+I*β    0        0 ] <- S
P(t) = [0          1-Ep       Ep       0 ] <- E
       [0          0          1-Ip     Ip] <- I
       [0          0          0        1 ] <- R
```

So given a state vector at time t and the transition matrix at time t, we can calculate the probability (or expected 
proportion) of being in the next state using an equation (called the Chapman-Komogrov equations) that state 
```p(t+1) = p(t)P(t)```. Here the vector p(t) is considered a 1x4 matrix and it is matrix multiplied with the 
4x4 transition matrix. 

Through this multiplication, it can be seen that the time spent as exposed follows a Geometric distribution with 
mean 1/Ep and the time spent as infected also follows Geometric with mean 1/Ip. Both of these Geometric distributions count the 
number of trials.

Thus, given a state at time 0 and some guesses for alpha, beta, infection probability and recovery probability, 
we can find the state of the system at any other future time.

We now consider substates in E and I. We assume there are m substates in E and n substates in I. We must first 
adjust our matrix P so instead of being a 4x4 matrix, it is of size (m+n+2)x(m+n+2). For example, when m=3, n=2 we have:
```
       [1-E*α+I*β  E*α+I*β    0        0        0        0        0 ] <- S
       [0          1-Ep       Ep       0        0        0        0 ] <- E substate 1
       [0          0          1-Ep     Ep       0        0        0 ] <- E substate 2
P(t) = [0          0          0        1-Ep     Ep       0        0 ] <- E substate 3
       [0          0          0        0        1-Ip     Ip       0 ] <- I substate 1
       [0          0          0        0        0        1-Ip     Ip] <- I substate 2
       [0          0          0        0        0        0        1 ] <- R
```

If we then have a "slide" along the main diagonal of 1-Ep and Ep along the upper diagonal in in the E superstate, 
we will have that the time spent in the E superstate follows a Negative Binomial distribution that counts 
the number of trials. The argument is similar for the I superstate. This is not desirable since this means the 
minimum number of days spent in E is m. We want a minimum of 0 days. To correct this, say for E, the row above the 
first substate in E must contain a Binomial PMF with parameters m and Ep. For example where ```C = (E\*α+I\*β)``` we have:
```
       [1-C  C(1-Ep)^3  3CEp(1-Ep)^2  3CEp^2(1-Ep)  CEp^3       0            0     ] <- S
       [0    1-Ep       Ep            0             0           0            0     ] <- E substate 1
       [0    0          1-Ep          Ep            0           0            0     ] <- E substate 2
P(t) = [0    0          0             1-Ep          Ep(1-Ip)^2  2EpIp(1-Ip)  EpIp^2] <- E substate 3
       [0    0          0             0             1-Ip        Ip           0     ] <- I substate 1
       [0    0          0             0             0           1-Ip         Ip    ] <- I substate 2
       [0    0          0             0             0           0            1     ] <- R
```

This then gets to the typical Negative Binomial that counts the number of failures.

We then pick values of Ep and Ip that give the desired mean time spent in each group. For COVID, the values that seem 
to match the best by manual tuning are m = 3, n = 7, mean time in E = 11 days, mean time in I = 20 days. 

This mean time in I matches with articles that say median time in I is about 2 weeks.
This mean time, however, in E is contrary to articles that say median time in E is 5 days. But this value of 11 days 
matches the South Korean data. South Korea has done so many tests, it is safe to say their reported numbers are the 
numbers in E instead of I. Then, aligning this with the number of recovered, the best value seems to be 11 days.

This is where the model stands now.


# Notes and estimation of alpha and beta

Alpha is the proportion of newly exposed per proportion of exposed per proportion of susceptible. So alpha can almost be thought of the number of people an exposed person infects per day.
Beta is the proportion of newly exposed per proportion of infected per proportion of susceptible. So beta can almost be thought of the number of people an infected person infects per day.

Beta was said to be a proportion of alpha for simplicity and to avoid some extreme results for both beta and alpha.

Estimation was done on a per country basis. There was a parametrisation of alpha and beta to be a Weibull function scaled 
by some constant and then another positive constant is added. The parameters for the Weibull and the constant were then estimated via minimising the MSE. 
The MSE is calculated by taking the model's predicted number of infected, subtracting that from the observed infected, 
squaring everything and summing over each time t. The minimising was done via the Nelder-Mead method.

# Pros for the model

- It is one of the simpler models and a discrete version is able to use matrices instead of differential equations which is (I think) a big computational advantage.
- The parameters are fairly easy to understand.
- It seems to match observed data and changes in real world events can be accounted for in alpha and beta.

# Shortcomings

- This model assumes that the population is homogeneous. This is usually not the case. The population usually live in different cities and people tend to have different habits and so infect others differently.
- This model does not account for deaths and so cannot show off an increased mortality rate when a large proportion of the population is infected at one time.
- We must assume an incorrect value for the mean exposed time in order to fit the data.
- The choice for Weibull in alpha and beta implicitly means that groups get progressively less contagious. i.e. Lock down only gets progressively harsher.
- This model does not account for some of the infected population having being 'imported' from other countries and assumes them to be infected due to internal infection dynamics.

# Data source
This dataset was the only one that I could find that has information on recovered.
If any one can find another that is potentially of better quality, please let me know.

https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases

# Inspiration
Here are some links to get started on the model.

https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3182455/
https://www.researchgate.net/publication/334695153_Complete_maximum_likelihood_estimation_for_SEIR_epidemic_models_theoretical_development

# Sources
https://www.worldometers.info/coronavirus/coronavirus-incubation-period/
https://www.worldometers.info/coronavirus/coronavirus-symptoms/
