#!/usr/bin/env python
# coding: utf-8

# In[3]:




from random import choice
from numpy import array,dot,random
import numpy as np


# In[10]:


#Mccullohpitt's Neuron

# for mimicing AND Gate
training_data = [
    (array([0,0]),0),
    (array([0,1]),0),
    (array([1,0]),0),
    (array([1,1]),1),
]


# In[22]:


step_function = lambda x : 0 if x < 2 else 1
#test 
step_function(1.5)


# In[88]:


w = random.rand(2)
w[0] = 1
w[1] = 1
#Mccullohpitt didnt use weights i just did it to generalize Mccullohpitt's and Rosenblats Neuron


# In[89]:


for x1,_ in training_data:
    result = dot(x1,w)
    print("{} : {} => {}".format(x1,result,step_function(result)))


# In[90]:


# we have to manually change the step function which is not right (how a neuron should work)
# Rosenblat 


# In[92]:


#Rosenblat's Neuron


# In[146]:


step_function = lambda x : 0 if x < 50 else 1
# random threshold

#OR Gate Mimicing

training_data = [
    (array([0,0,0]),0),
    (array([0,1,1]),1),
    (array([1,0,1]),1),
    (array([1,1,1]),0),
]

# [x1,x2,b] * [m1 /n * m2 /n * m3 /n]


# In[147]:


w = random.rand(3)
w


# In[148]:


b = 0.1
eta = 10 #learning rate
errors = []
n = 1000 #steps


# In[149]:


for i in range(n):
    x , expected = choice(training_data)
    
    result = dot(x,w)
    # [x1,x2,b] * [w[0],w[1],w[3]]
    #print(result)
    error = expected - step_function(result)
   
    errors.append(error)
    w  = w + eta*error*x
    


# In[150]:


for x , _ in training_data:
    result = dot(x,w)
    print("{} : {} => {}".format(x[:3],result,step_function(result)))


# In[119]:




