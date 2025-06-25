import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from sigmoid_function import sigmoid
from Cost_function_for_logistic_regression import compute_cost_logistic_regression

def compute_gradient_logistic(x, y, w, b):
    """
    computes the gradient for logistic regression
    find w and b that minimizes the cost function

    :param x: data m examples with n features
    :param y: target values
    :param w: model parameter
    :param b: model parameter
    :return: dj_dw: the gradient of the cost function of the parameter w
             dj_db: the gradient of the cost function of the parameter b
    """
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_i = sigmoid(np.dot(x[i],w)+b)
        err_i = f_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * x[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


def gradient_descent_logistic(x, y, w_in, b_in, alpha, num_iterations):
    """
    performs patch of gradient descent

    :param x: data, m examples with n features
    :param y: target values
    :param w_in: Initial values of model parameter
    :param b_in: Initial value of model parameter
    :param alpha: Learning rate
    :param num_iterations: number of iterations to run gradient descent
    :return:
        w: updated values of parameters
        b: updated value of parameter
        j_history: history for visualization
    """

    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i<100000:
            j_history.append(compute_cost_logistic_regression(x, y, w, b))

        if i%math.ceil(num_iterations/10)==0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]} ")

    return w, b, j_history

if __name__ == "__main__":
    #example training data:
    x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='red', marker='x', label='zeros class')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='blue', marker='o', label='ones class')
    plt.grid(True)
    plt.show()

    #example to test gradient function:
    #expected: dj_dw: [0.498333393278696, 0.49883942983996693] ,,,, dj_db: 0.49861806546328574
    w_tmp = np.array([2., 3.])
    b_tmp = 1.
    dj_dw_tmp, dj_db_tmp= compute_gradient_logistic(x_train, y_train, w_tmp, b_tmp)
    print(f"dj_db: {dj_db_tmp}")
    print(f"dj_dw: {dj_dw_tmp.tolist()}")

    #example to test gradient descent function:
    w_tmp = np.zeros_like(x_train[0])
    b_tmp = 0.
    alpha = 0.1
    iterations = 10000

    w_out, b_out = gradient_descent_logistic(x_train,y_train,w_tmp,b_tmp,alpha,iterations)

    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")