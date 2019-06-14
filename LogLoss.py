import numpy as np
from matplotlib import pyplot as plt

InputData = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

TargetData = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def loss(a, y):
    if y == 1:
        return -np.log(a)
    else:
        return -np.log(1-a)

def loss_d(a, y):
    if y == 1:
        return -1/a
    else:
        return 1/(1-a)

w1 = np.zeros((3, 2))
b1 = np.random.randn(3, 1)

w2 = np.zeros((1, 3))
b2 = np.random.randn()

iterations = 100000
lr = 0.1
costlist = []

for i in range(iterations):
    random = np.random.choice(len(InputData))
    x = InputData[random].reshape(2, 1)

    if i % 100 == 0:
        c = 0
        for j in range(len(InputData)):
            z1 = np.dot(w1, InputData[j].reshape(2, 1)) + b1
            a1 = sigmoid(z1)

            z2 = np.dot(w2, a1) + b2
            a2 = sigmoid(z2)
            c += float(loss(a2, TargetData[j]))
        costlist.append(c)
    
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    #backprop
    dcda2 = loss_d(a2, TargetData[random])
    da2dz2 = sigmoid_p(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = sigmoid_p(z1)
    dz1dw1 = InputData[random].reshape(1, 2)

    w2 = (w2.T - lr * dcda2 * da2dz2 * dz2dw2).T
    b2 = b2 - lr * dcda2 * da2dz2

    w1 = w1 - np.dot((lr * dcda2 * da2dz2 * w2.T * da1dz1), dz1dw1)
    b1 = b1 - lr * dcda2 * da2dz2 * w2.T * da1dz1

plt.plot(costlist)
plt.show()

for j in range(len(InputData)):
    z1 = np.dot(w1, InputData[j].reshape(2, 1)) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    cost = float(loss(a2, TargetData[j]))
    print("Prediction: ", a2)
    print("Cost: ", cost)
    print("\n")
