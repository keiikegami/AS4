import numpy as np
from scipy.stats import binom

n = 25

iteration = 100
values = np.ones((iteration, 3))
initial_value = [1/3, 1/3, 1/3]
values[0, :] = initial_value

for ite in range(1, iteration):
    q_a = (2*values[ite-1, 2]) / (2*values[ite-1, 2] + values[ite-1, 0])
    q_b = (2*values[ite-1, 2]) / (2*values[ite-1, 2] + values[ite-1, 1])
    under = sum([binom.pmf(i, n, q_a) * binom.pmf(j, n, q_b) * (3*n - i) for i in range(n+1) for j in range(n+1)])
    upper = sum([binom.pmf(i, n, q_a) * binom.pmf(j, n, q_b) * (2*n + i + j) for i in range(n+1) for j in range(n+1)])
    values[ite, 0] = 1/(2+(upper/under))
    values[ite, 1] = 1/(2+(upper/under))
    values[ite, 2] = 1 - 2*values[ite, 0]
    
