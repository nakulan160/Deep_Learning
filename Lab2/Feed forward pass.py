import numpy as np


def calcualte_z(x,w,bias):
    return np.dot(x,w)+np.sum(bias)

def activation(z):
    return(np.maximum(0,z))

x=np.array([21,227,93,42,57])
w=np.array([567,40,34,2,13])
bias=np.array([0,4,6,8,2])
z1=calcualte_z(x,w,bias)
y_pred=activation(z1)
print("y_pred",y_pred)


