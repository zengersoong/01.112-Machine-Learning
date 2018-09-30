import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

csv = 'linear.csv'
data = np.genfromtxt(csv, delimiter=',')

regr = linear_model.SGDRegressor(fit_intercept=False)
epochs = 1000

for i in range(epochs):
    edited = np.random.permutation(data)
    edited = edited[5:].copy()
    X = edited[:, 1:]
    Y = edited[:, 0]
    regr.partial_fit(X, Y)


print('Coefficients: \n', regr.coef)