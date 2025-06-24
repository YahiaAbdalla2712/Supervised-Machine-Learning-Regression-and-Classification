import numpy as np
import matplotlib.pyplot as plt
from sigmoid_function import sigmoid

def compute_cost_logistic_regression(x, y, w, b):
    """
    compute cost for logistic regression

    :param x: array x of (m,n): data, m with n features
    :param y: array y: target values
    :param w: model parameter
    :param b: model parameter
    :return: cost value
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_i) - (1-y[i])*np.log(1-f_i)

    cost = cost/m
    return cost


if __name__ == "__main__":
    #example datasets:
    x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    #ploting train data:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='red', marker='x', label='zeros class')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='blue', marker='o', label='ones class')
    plt.grid(True)
    plt.show()

    w_tmp = np.array([1, 1])
    b_tmp = -3
    print(compute_cost_logistic_regression(x_train, y_train, w_tmp, b_tmp))
