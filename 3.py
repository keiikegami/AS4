import numpy as np
from scipy.special import p_roots
from scipy.special import u_roots
from scipy.special import t_roots

def f(x):
    return (x**9) / np.sqrt(x**2 + 1)

def GL(m, a, b):
    points = p_roots(m)[0]
    weights = p_roots(m)[1]
    return (b-a)*sum([weights[i]*f((b-a)*points[i]/2 + (b+a)/2)  for i in range(len(points))])/2

def Cheby1(m, a, b):
    points = t_roots(m)[0]
    weights = t_roots(m)[1]
    return (b-a)*sum([weights[i]*np.sqrt(1-(points[i])**2) * f((b-a)*points[i]/2 + (b+a)/2)  for i in range(len(points))])/2

def Cheby2(m, a, b):
    points = u_roots(m)[0]
    weights = u_roots(m)[1]
    return (b-a)*sum([(weights[i]/np.sqrt(1-(points[i])**2)) * f((b-a)*points[i]/2 + (b+a)/2)  for i in range(len(points))])/2

a = [5, 10]

for m in a:
    # approximation results
    print(GL(m))
    print(Cheby1(m))
    print(Cheby2(m))
    
    # nodes
    print(p_roots(m)[0])
    print(t_roots(m)[0])
    print(u_roots(m)[0])
    
    # weights
    print(p_roots(m)[1])
    print(t_roots(m)[1])
    print(u_roots(m)[1])