#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
from sklearn.decomposition import NMF
# Non-Negative matrix Factorization
# By using the decomposition function from sklearn library
np.set_printoptions(precision=3)

X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 2]])
model = NMF(n_components=1, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_



A = np.dot(W,H)
print("Matrix A:")
print(A)

Distance = ((A[0][0]-0)**2 +( A[0][1]-1)**2 + (A[1][0]-1)**2 + (A[1][2]-1)**2 +(A[2][1]-1)**2 + (A[2][2]-2)**2)


print("Matrix U:")
print(W)

matV = (H.T)
print("Matrix V:")
print(matV)
# Answer is a local minima
# Using NMF provides better intrpretation of lower rank approximation such as this question
print("Distance from original incompleted matrix is "+ str(Distance))


# In[32]:


#Question 2 RBF SVM to classify the datas into two classes
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
csv = 'kernel.csv'
data =np.genfromtxt(csv,delimiter=",")
X = data[:,1:] #This is an array of array, data point has two feature value x1,x2
Y = data[:,0]
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[2]:


# Using sklearn.svm.SVC module in the Python Scikit-learn package to train a kernel 
# support vector machine via the radial basis kernel.
# Remember to set the gamma to 0.5 and kernel to rbf when initialising the object
# (2a) Support Vector Machine
from sklearn import svm

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.5,
    kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)


clf.fit(X,Y)# Train the SVM
clf.predict([[-3, -3]]) # Testing the predictor


# In[3]:


#Radial Basis function (RBF)
# RBF kernel can choose a non-linear decision boundary
# Side notes
# 1.
# The dataset without a linear as can be seen above, 
# we can use RBF to automatically decide a non-linear decision boundary.
# 2.
# Note that regularization parameter "The degree in which we allow violations", also know as C parameter in sklearn library tells the SVM optimization 
# how much you want to avoid misclassifying each training example.
# 2.a
# For large C, the optimization will choose a smaller-margin hyperplane
# Resulting in getting more training points to be classified correctly.
# However may also result in overfitting problems.
# 2.b
# For smaller C, the optimization wil 
# This might result in a few misclassification of points.
# This may or may not be bad, as the points may be outliers data points.
# We should shrink to accomadate them anyway.
# 3.a
# Gamma parameter
# The gamma parameter defines how far the influence of a single training
# example reaches.
# 3.b
# Low Gamma means that points that are far away from the "street boundaries" are considered
# in calculation.
# 3.c
# High Gamma parameter means that only points close to the plausible "street bondaries" are considered.


# In[4]:


# Evaluate the kernel SVM’s decision function. You may use the decision function
# method in sklearn.svm.SVC. You should write a function that takes coordinates
# x1, x2 of a point x ∈ R

# decision_function” that computes the signed distance of a point from the boundary. 
# A negative value indicates class 0 and a positive value would indicate class 1. 
# Also, a value close to 0 would indicate that the point is close to the boundary.
def decision(x1,x2,clf):
    x = np.array([x1,x2])
    return clf.decision_function([x])

decision(3.8,3.3,clf)


# In[5]:


# Not in assignment just done for fun
# Confidence measure in the classification, Class A or Class B 
# Set Probability to True.
clf2 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.5,
    kernel='rbf', max_iter=-1, probability=True,
    random_state=None, shrinking=True, tol=0.001, verbose=False)

clf2.fit(X,Y)

def confidenceMeasure(x1,x2,clf):
    x = np.array([[x1,x2]])
    return clf.predict_proba(x)
confidenceMeasure(3.8,3.3,clf2)


# In[6]:


vdecision = np.vectorize(decision,excluded=[2])
x1list = np.linspace(-8.0,8.0,100)
x2list = np.linspace(-8.0,8.0,100)
X1, X2 = np.meshgrid(x1list,x2list)
Z = vdecision(X1,X2,clf)
cp = plt.contourf(X1,X2,Z)
plt.colorbar(cp)
plt.scatter(X[:,0],X[:,1],c=Y,cmap='gray')
plt.show()


# In[7]:


# Question 3 unsupervised Deep Learning
# Sparse autoencoder


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b as minimize
from utils import normalize, tile_raster_images, sigmoid
from utils import ravelParameters, unravelParameters
from utils import initializeParameters
from utils import computeNumericalGradient
nV = 8*8 # number of visible units
nH = 25 # number of hidden units
dW = 0.0001 # weight decay term
sW = 3 # sparsity penalty term
npy = 'images.npy'
X = normalize(np.load(npy))
plt.imshow(tile_raster_images(X=X,
img_shape=(8,8),tile_shape=(5,5),
tile_spacing=(1,1)),cmap='gray')
plt.show()


# In[9]:


# We implement the function which computes the cost and the gradient of the sparse
# autoencoder. This function will be passed to an optimization engine, together with
# the theta vector that contains the current state of all the model parameters. The
# first step of the function is therefore to unpack the theta vector into W1, W2, b1, b2.
# Some of the other steps are provided in the template below.
nV = 8*8 # number of visible units
nH = 25 # number of hidden units
dW = 0.0001 # weight decay term
sW = 3 # sparsity penalty term
def sparseAutoencoderCost(theta,nV,nH,dW,sW,X):
    W1,W2,b1,b2 = unravelParameters(theta,nH,nV)
    n = X.shape[0] # 1d shaped array.
    z2 =np.dot(X,W1)+np.dot(np.ones(b1.shape[1]),b1.T)
    a2 = sigmoid(z2)
    z3 =np.dot(a2,W2)+np.dot(np.ones(b2.shape[1]),b2.T)
    a3 = sigmoid(z3)
    eps = a3-X
    loss =(norm(eps)**2)/(2*n)
    decay =((norm(W1)**2)+(norm(W2)**2))/2
    # Compute sparsity terms and total cost
    rho = 0.01
    a2mean = np.mean(a2,axis=0).reshape(nH,1)
    kl = np.sum(rho*np.log(rho/a2mean)+    (1-rho)*np.log((1-rho)/(1-a2mean)))
    dkl = -rho/a2mean+(1-rho)/(1-a2mean)
    cost = loss+dW*decay+sW*kl
    d3 =eps*a3*(1-a3)
    d2 = (sW*dkl.T+np.dot(d3,W2.T))*a2*(1-a2)
    W1grad =np.dot(X.T,d2)/n +dW*W1
    W2grad =np.dot(a2.T,d3)/n +dW*W2
    b1grad =np.dot(d2.T,np.ones(n))/n
    b2grad =np.dot(d3.T,np.ones(n))/n
    grad = ravelParameters(W1grad,W2grad,b1grad,b2grad)
    print(' .',end="")
    return cost,grad


theta = initializeParameters(nH,nV)
cost,grad = sparseAutoencoderCost(theta,nV,nH,dW,sW,X)

print(cost)
print(grad)


# In[10]:


print('\nComparing numerical gradient with backprop gradient')
num_coords = 5
indices = np.random.choice(theta.size,num_coords,replace=False)
numgrad = computeNumericalGradient(lambda t:
sparseAutoencoderCost(t,nV,nH,dW,sW,X)[0],theta,indices)
subnumgrad = numgrad[indices]
subgrad = grad[indices]
diff = norm(subnumgrad-subgrad)/norm(subnumgrad+subgrad)
print('\n',np.array([subnumgrad,subgrad]).T)
print('The relative difference is',diff)


# In[11]:


print('\nTraining neural network')
theta = initializeParameters(nH,nV)
opttheta,cost,messages = minimize(sparseAutoencoderCost,
theta,fprime=None,maxiter=400,args=(nV,nH,dW,sW,X))
W1,W2,b1,b2 = unravelParameters(opttheta,nH,nV)
plt.imshow(tile_raster_images(X=W1.T,
img_shape=(8,8),tile_shape=(5,5),
tile_spacing=(1,1)),cmap='gray')
plt.show()


# In[34]:


# Question 4
# (a) In the slides of Lecture 8 (generative model), we discussed using MLE to calculate
# the probability of word w so that the probability that a document is generated is
# maximized. Please take a look at slides 13 and 14, and proof how θw is derived.
# Does it matter whether W denotes the collection of words in document S or the
# collection of words in the dictionary?
# (b) In slide 15 of Lecture 8, we discussed how to estimate the parameters (µ and σ)
# so that the data samples are generated with a maximum probability under the
# spherical Gaussian assumption. Please demonstrate how µ and σ are derived based
# on MLE.
#These is on PDF format written.


# In[ ]:




