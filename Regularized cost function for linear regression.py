import numpy as np
import matplotlib.pyplot as plt

def compute_cost_for_linear_regression_with_regularization(x, y, w, b, lambda_=1):
    """
    computes the cost over all examples with regularization feature

    :param x: data, m examples with n features
    :param y: target values
    :param w: model parameter
    :param b: model parameter
    :param lambda_: regularization parameter
    :return:
        total_cost: cost
    """

    m = x.shape[0]
    n = len(w)
    cost = 0
    for i in range(m):
        f_i = np.dot(x[i], w)+b
        cost = cost + (f_i - y[i])**2
    cost = cost / (2*m)

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost

    total_cost = cost + reg_cost
    return total_cost

#example:
np.random.seed(1)
x_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(x_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost = compute_cost_for_linear_regression_with_regularization(x_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost)

