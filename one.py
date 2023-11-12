#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# **1. Tensorflow**

# In[2]:


import tensorflow as tf


# In[3]:


print(tf.__version__)


# **2. Keras**

# In[4]:


from keras import datasets
# Load MNIST datasets from keras
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


# In[5]:


train_images.shape


# In[6]:


test_images.shape


# **3. Theano**

# In[7]:


get_ipython().system('pip install Theano')


# In[8]:


import theano.tensor as T
from theano import function


# In[ ]:


# Declaring 2 variables
x = T.dscalar('x')
y = T.dscalar('y')


# In[ ]:


# Summing up the 2 numbers
z = x + y


# In[ ]:


# Converting it to a callable object so that it takes matrix as parameters
f = function([x, y], z)


# In[ ]:


f(5, 7)


# **4. PyTorch**

# In[ ]:


get_ipython().system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115')


# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


print(torch.__version__)


# In[ ]:


torch.cuda.is_available()

