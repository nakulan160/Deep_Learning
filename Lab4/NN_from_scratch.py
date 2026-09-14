import numpy as np
import matplotlib.pyplot as plt

x=np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
],dtype=float)

y=np.array([
    [0],
    [1],
    [1],
    [0]
],dtype=float)

np.random.seed(1)

w=np.random.randn(3,1)*0.5
b=0.0

lr=0.5
losses=[]
ws=[]
grads=[]

def sig(x):
    return 1/(1+np.exp(-x))

for i in range(1000):

    z=np.dot(x,w)+b
    a=sig(z)

    loss=np.mean((y-a)**2)
    losses.append(loss)

    dz=(a-y)*a*(1-a)

    dw=np.dot(x.T,dz)
    db=np.sum(dz)

    ws.append(w.copy())
    grads.append(dw.copy())

    w-=lr*dw
    b-=lr*db

    if i%100==0:
        print(i,loss)

z=np.dot(x,w)+b
a=sig(z)

print("Predictions:")
for i in range(len(x)):
    p=1 if a[i][0]>=0.5 else 0
    print(x[i].astype(int),round(a[i][0],4),p)

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

ws=np.array(ws).reshape(1000,3)
grads=np.array(grads).reshape(1000,3)

plt.plot(ws[:,0],label="w1")
plt.plot(ws[:,1],label="w2")
plt.plot(ws[:,2],label="w3")
plt.xlabel("Iterations")
plt.ylabel("Weights")
plt.legend()
plt.show()

plt.plot(grads[:,0],label="dw1")
plt.plot(grads[:,1],label="dw2")
plt.plot(grads[:,2],label="dw3")
plt.xlabel("Iterations")
plt.ylabel("Gradients")
plt.legend()
plt.show()