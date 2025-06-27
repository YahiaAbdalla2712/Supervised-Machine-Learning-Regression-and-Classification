import numpy as np
from sigmoid_function import sigmoid

def compute_gradient_logistic_reg(x, y, w, b, lambda_):
    """
    computes the gradient descent for regularized logistic regression

    :param x: data, m examples with n features
    :param y: target values
    :param w: model parameter
    :param b: model parameter
    :param lambda_: regularization parameter
    :return:
        dj_dw: the gradient cost with respect to the parameter w
        dj_db: the gradient cost with respect to the parameter b
    """

    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_i = sigmoid((np.dot(x[i], w))+b)
        error_i = f_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error_i * x[i, j]
        dj_db = dj_db + error_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_dw, dj_db


#example:
np.random.seed(1)
x = np.random.rand(5, 3)
y = np.array([0, 1, 0, 1, 0])
w = np.random.rand(x.shape[1])
b = 0.5
lambda_ = 0.7
dj_dw, dj_db = compute_gradient_logistic_reg(x, y, w, b, lambda_)

print(f"dj_db: {dj_db}")
print(f"Regularized dj_dw:\n {dj_dw.tolist()}")