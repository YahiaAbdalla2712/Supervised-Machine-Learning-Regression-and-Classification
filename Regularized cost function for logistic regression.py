import numpy as np
from sigmoid_function import sigmoid

def compute_cost_for_logistic_regression_with_regularization(x, y, w, b, lambda_=1):
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

    m, n = x.shape
    cost = 0
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_i) - (1-y[i])*np.log(1-f_i)
    cost = cost/m

    regularized_cost = 0
    for j in range(n):
        regularized_cost += (w[j]**2)
    regularized_cost = (lambda_/(2*m)) * regularized_cost

    total_cost = cost + regularized_cost
    return total_cost


#example:
np.random.seed(1)
x = np.random.rand(5, 6)
y = np.array([0, 1, 0, 1, 0])
w = np.random.rand(x.shape[1]).reshape(-1,)-0.5
b = 0.5
lambda_ = 0.7
cost = compute_cost_for_logistic_regression_with_regularization(x, y, w, b, lambda_)

print("Regularized cost:", cost)