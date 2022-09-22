#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[104]:


X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y=y.reshape(y.shape[0],1)

print ("dimension de X:", X.shape)
print ("dimension de y:", y.shape)
plt.scatter(X[:,0],X[:,1], c=y, cmap = "summer")
plt.show()


# In[105]:


def initialisation (X):
    W = np.random.randn (X.shape[1], 1)
    b = np.random.randn (1)
    return (W, b)
    


# In[106]:


initialisation(X)


# In[107]:


def model (X, W, b):
    Z = X.dot(W)+ b
    A= 1 / (1+ np.exp(-Z))
    return A


# In[108]:


#fonction Model
A = model(X, W, b)
A.shape


# In[109]:


def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1-y) * np.log(1-A)) 


# In[110]:


log_loss(A, y)


# In[49]:


#Fonction gradients
def gradients(A, X, y):
    dw = 1 / len(y) * np.dot(X.T, A -y)
    db = 1 / len(y) * np.sum(A-y)
    return (dw,db)


# In[50]:


dw, db = gradients(A,X,y)

db.shape


# In[51]:


def update(dw, db, W, b, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)
    


# In[65]:


#effectuer les futures prédictions

def predict(X, W, b):
    A = model(X, W, b)
    print (A) #Pour imprimer la probabilité associé à une plante
    return A>=0.5


# In[67]:


from sklearn.metrics import accuracy_score


# In[103]:


def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    #initialisation w,b
    W, b = initialisation (X)
    
    Loss = []
    
    #creation de la boucle d'apprentissage
    for i in range (n_iter):
            A = model(X, W, b)
            #creation d'une variable pour l'evolution de notre coût
            Loss.append (log_loss (A, y)) # Rajouter de la valeur du coût qui est calculé pour l'iteration en cour
            dw, db = gradients(A, X, y)
            W, b = update (dw, db, W, b, learning_rate)
            
    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred)) #compare les données de prédictions avec les y_pred
            
    plt.plot(Loss)
    plt.show
    
    return (W, b)
            


# In[111]:


W, b = artificial_neuron(X,y)


# In[112]:


new_plant =np.array([2,1])
x0 = np.linspace(-1, 4, 100)
x1 = ( -W[0] * x0 - b ) / W[1]

plt.scatter(X[:,0],X[:,1], c=y, cmap = "summer")
plt.scatter(new_plant[0], new_plant[1], c= "r")
plt.plot (x0, x1, c="orange", lw=3)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




