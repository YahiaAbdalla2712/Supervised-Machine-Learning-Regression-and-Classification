import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """sigmoid function used in logistic regression maps all input values to values between 0 and 1"""

    g = 1/(1+np.exp(-z))

    return g


#usage example:

#array of evenly spaced values between -10 and 10
z = np.arange(-10,11)

y = sigmoid(z)

print("input z values")
print(z)
print("output y values")
print(y)

#plot z vs sigmoid(z):

fig, ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z, y, c="b")
plt.show()
