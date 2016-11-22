import numpy as np
from scipy.special import eval_hermite
from scipy.special import hermite
from scipy.special import h_roots
from scipy.special import he_roots
from scipy.stats import norm

def GH(m):
    points = h_roots(m)[0]
    weights = h_roots(m)[1]
    return sum([weights[i] * np.exp(points[i]**2) * norm.pdf(points[i]) for i in range(len(points))])

def GH_weights(m):
    points = h_roots(m)[0]
    weights =  h_roots(m)[1]
    return [weights[i] * np.exp(points[i]**2) for i in range(len(points))]

a = [5, 10, 20, 30]

# approximation result
for m in a:
    print(GH(m))

# weights
for m in a:
    print(GH_weights(m))

# points
for m in a:
    print(h_roots(m)[0])
