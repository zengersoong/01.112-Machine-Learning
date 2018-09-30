#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
csv = 'linear.csv'
data = np.genfromtxt(csv,delimiter=',')
X = data[:,1:]
Y = data[:,0]


# In[4]:


import theano
import theano.tensor as T
d = X.shape[1] # dimension of feature vectors
n = X.shape[0] # number of training samples
learn_rate = 0.5 # learning rate for gradient descent


# In[5]:


x = T.matrix(name='x') # feature matrix
y = T.vector(name='y') # response vector
w = theano.shared(np.zeros((d,1)),name='w') # model parameters


# In[6]:


risk = T.sum((T.dot(x,w).T - y)**2)/2/n # empirical risk
grad_risk = T.grad(risk, wrt=w) # gradient of the risk


# In[7]:


train_model = theano.function(inputs=[],
outputs=risk,
updates=[(w, w-learn_rate*grad_risk)],
givens={x:X, y:Y})


# In[9]:


n_steps = 50
for i in range(n_steps):
    print(train_model())
print(w.get_value())


# In[ ]:




