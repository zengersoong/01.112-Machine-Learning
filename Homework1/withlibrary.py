import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

csv = 'linear.csv'
data = np.genfromtxt(csv, delimiter=',')
X = data[:, 1:]
Y = data[:, 0]

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(X, Y)

print('Coefficients: \n', regr.coef_)