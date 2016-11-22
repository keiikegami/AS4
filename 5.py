# data generate
import numpy as np

I = 100
J = 2
beta_0 = -1
beta_1 = 1
sigma_u = 0.5
sigma_eps = 1

np.random.seed(12345)

value_x = np.random.normal(0, 1, (I, J))
u = np.random.normal(0, sigma_u, I)
value_u = np.ones((I, J))
value_u[:, 0] = u
value_u[:, 1] = u
value_eps = np.random.normal(0, sigma_eps, (I, J))
value_y = beta_0 + beta_1 * value_x + value_u + value_eps


# implement EM algorithm
# para = (beta_0, beta_1, sigma_eps, sigma_u)
# sigma is variance
iteration = 500
XY = sum(sum(value_y * value_x))
X = sum(sum(value_x))
Y = sum(sum(value_y))
X_2 = sum(sum(value_x * value_x))

def xE(E_t):
    return sum(sum(value_x * np.array([E_t, E_t]).T))

def eps_right(E_t, beta0, beta1):
    return (sum(sum((value_y - beta0-beta1*value_x)**2)) + J * sum(E_t**2) - 2*sum(sum((value_y - beta0 - beta1*value_x) * np.array([E_t, E_t]).T)))/(I*J)

initial = [0, 0, 5, 5]
E_t = np.ones(I)
estimation = np.ones((iteration, 4))
estimation[0, :] = initial

for ite in range(1,iteration):
    u = estimation[ite-1, 3]
    eps = estimation[ite-1, 2]
    beta0 = estimation[ite-1, 0]
    beta1 = estimation[ite-1, 1]
    for i in range(I):
        E_t[i] = u*sum([value_y[i, j] - beta0 - beta1*value_x[i, j] for j in range(J)]) / (J*u + eps)
    V_t = eps*u/(J*u + eps)
    estimation[ite, 1] = (I*J *XY - Y*X - J * (I*xE(E_t) - sum(E_t)*X))/(I*J*X_2 - X**2)
    estimation[ite, 0] = (Y-beta1*X - J* sum(E_t))/(I*J)
    estimation[ite, 2] = V_t + eps_right(E_t, estimation[ite-1, 0], estimation[ite-1, 1])
    estimation[ite, 3] = V_t + sum(E_t**2)/I
    
# change the last variance into standard deviation
estimation[:, 2] = np.sqrt(estimation[:, 2])
estimation[:, 3] = np.sqrt(estimation[:, 3])

# compute the bias and std
bias = np.average(estimation, axis = 0) - [-1, 1, 1, 0.5]
std = np.std(estimation, axis = 0)
print("bias :", bias)
print("std :", std)