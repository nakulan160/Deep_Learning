
import numpy as np

x=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
y=np.array([[0],[1],[1],[0]],dtype=float)

def sig(x):
    return 1/(1+np.exp(-x))

def dsig(x):
    s=sig(x)
    return s*(1-s)

np.random.seed(42)

w1=np.random.randn(2,2)*0.5
b1=np.zeros((1,2))

w2=np.random.randn(2,1)*0.5
b2=np.zeros((1,1))

lr=0.5
epochs=10000

for i in range(epochs):
    z1=np.dot(x,w1)+b1
    a1=sig(z1)

    z2=np.dot(a1,w2)+b2
    a2=sig(z2)

    loss=np.mean((y-a2)**2)

    dz2=(a2-y)*dsig(z2)
    dw2=np.dot(a1.T,dz2)
    db2=np.sum(dz2,axis=0,keepdims=True)

    da1=np.dot(dz2,w2.T)
    dz1=da1*dsig(z1)
    dw1=np.dot(x.T,dz1)
    db1=np.sum(dz1,axis=0,keepdims=True)

    w2-=lr*dw2
    b2-=lr*db2
    w1-=lr*dw1
    b1-=lr*db1

    if i%1000==0:
        print(i,loss)

z1=np.dot(x,w1)+b1
a1=sig(z1)
z2=np.dot(a1,w2)+b2
a2=sig(z2)

print("Predictions:")

for i in range(4):
    p=1 if a2[i][0]>=0.5 else 0
    print(x[i].astype(int),round(a2[i][0],4),p)

