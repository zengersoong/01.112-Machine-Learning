import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x1,x2,x3,x4, y, x1_current=0,x2_current=0,x3_current=0, x4_current=0,epochs=1000, learning_rate=0.01):
    N = float(len(y))
    plotpt = []
    for i in range(epochs):
        y_current = (x1_current * x1) + (x2_current*x2) + (x3_current*x3) + (x4_current*x4)
        cost = sum([data**2 for data in (y-y_current)]) / N

        x1_gradient = -(2/N) * sum(x1 * (y - y_current))
        x2_gradient = -(2/N) * sum(x2 * (y - y_current))
        x3_gradient = -(2/N) * sum(x3 * (y - y_current))
        x4_gradient = -(2/N) * sum(x4 * (y - y_current))

        x1_current = x1_current - (learning_rate * x1_gradient)
        x2_current = x2_current - (learning_rate * x2_gradient)
        x3_current = x3_current - (learning_rate * x3_gradient)
        x4_current = x4_current - (learning_rate * x4_gradient)

        plotpt.append(cost)
    plt.plot(plotpt)
    plt.show()
    return x1_current, x2_current, x3_current, x4_current, cost

csv = 'linear.csv'
csv = np.genfromtxt (csv, delimiter=",")
y = csv[:,0]
x1 = csv[:,1]
x2 = csv[:,2]
x3 = csv[:,3]
x4 = csv[:,4]
print(linear_regression(x1,x2,x3,x4,y))