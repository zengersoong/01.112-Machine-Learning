#Done in Collaboration with Chang Jun Qing 1002088 - he forget anotate this
import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b as minimize

# Qn 1(a)
csv = 'linear.csv'
data = np.genfromtxt(csv,delimiter=',')
X = data[:,1:]
Y = data[:,0]

vX = X[0:10]
tX = X[10:]

vY = Y[0:10]
tY = Y[10:]

print("==Shapes== \nvX:{vx} \ntX:{tx} \nvY:{vy} \ntY:{ty}".format(vx=vX.shape, tx=tX.shape, vy=vY.shape, ty=tY.shape))

# ====================================================================================
# Qn 1(b)

steps=50
learn_rate=0.5
penalty = 0.15
d = tX.shape[1] # dimension of feature vectors
n = tX.shape[0] # number of training samples
x = T.matrix(name='x') # feature matrix
y = T.vector(name='y') # response vector
w = theano.shared(np.zeros((d,1)),name='w') # model parameters
risk = T.sum((T.dot(x,w).T - y)**2)/2/n + T.sum(T.dot(penalty/2,w**2))# empirical risk
grad_risk = T.grad(risk, wrt=w) # gradient of the risk
train_model = theano.function(inputs=[],
                            outputs=risk,
                            updates=[(w, w-learn_rate*grad_risk)],
                            givens={x:tX, y:tY})
for i in range(steps):
    train_model()
print(w.get_value())

# ====================================================================================
# Qn 1(c)
# TODO
# def costgrad(coefficient,x,y,l):
#     cost = np.sum((y-np.sum(np.dot(x,coefficient)))**2) + (l/2)*np.sum(coefficient**2)
#     grad = 2*np.multiply(np.transpose(x),(np.dot(x,coefficient)-y))+ np.dot(coefficient
#     print(grad)
#     return cost,grad
# l = 0.15
# coefficient = np.random.randn(4)
# x = tX        #data to pass into costgrad
# y = tY   #parameter to be optimized
# optcoeff,cost,messages = minimize(costgrad,coefficient,args=[x,y,l])
# print(optcoeff)



def costgrad(w,x,y):
	n = x.shape[0]
	lamda = 0.15
	emp_risk = np.sum((np.dot(x,w).flatten() - y) **2)/2/n
	reg_risk = np.sum(w[:-1] **2 ) * lamda/2
	risk = emp_risk +reg_risk
	grad = (lamda *np.append(w[:-1],0)) + 1/n * np.dot(np.dot(np.transpose(x),x),w) - 1/n * np.dot(np.transpose(x),y)
	return risk,grad
w = np.zeros((d,1))
optx, cost, messages = minimize(costgrad,w,args =[tX,tY])
print(optx)

# ====================================================================================
# Qn 1(d)
def ridge_regression(X, Y, reg_penalty=0.15):
    learn_rate=0.5 
    d = X.shape[1] # dimension of feature vectors
    n = X.shape[0] # number of training samples
    x = T.matrix(name='x') # feature matrix
    y = T.vector(name='y') # response vector
    w = theano.shared(np.zeros((d,1)),name='w') # model parameters
    risk = T.sum((T.dot(x,w).T - y)**2)/2/n + T.sum(T.dot(reg_penalty/2,w**2))# empirical risk
    grad_risk = T.grad(risk, wrt=w) # gradient of the risk
    train_model = theano.function(inputs=[],
                                outputs=risk,
                                updates=[(w, w-learn_rate*grad_risk)],
                                givens={x:X, y:Y})
    for i in range(steps):
        train_model()
    return(w.get_value())
        
w = ridge_regression(tX, tY, 0.15)
print(w)


# ====================================================================================
# Qn 1(e)


tn = tX.shape[0]
vn = vX.shape[0]
tloss = []
vloss = []
index = -np.arange(0,5,0.1)
for i in index:
    w = ridge_regression(tX,tY,10**i)
    tloss = tloss+[np.sum((np.dot(tX,w)-tY)**2)/tn/2]
    vloss = vloss+[np.sum((np.dot(vX,w)-vY)**2)/vn/2]

import matplotlib.pyplot as plt
plt.plot(index,np.log(tloss),'r')
plt.plot(index,np.log(vloss),'b')
plt.show()
print("ans:lambda =0.34119")
# x = -0.467
# lambda = 10**-0.467 = 0.34119
