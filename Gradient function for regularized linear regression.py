import numpy as np

def compute_gradient_linear_reg(x, y, w, b, lambda_):
    """
    Computes the gradient descent for linear regression

    :param x: data, m examples with n features
    :param y: target values
    :param w: model parameter
    :param b: model parameter
    :param lambda_: regularization parameter
    :return:
        dj_dw: the gradient of the cost with respect to parameter w
        dj_db: the gradient of the cost with respect to parameter b
    """

    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error*x[i, j]
        dj_db = dj_db + error

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
dj_dw, dj_db = compute_gradient_linear_reg(x, y, w, b, lambda_)

print(f"dj_db: {dj_db}")
print(f"regularized dj_dw:\n {dj_dw.tolist()}")
