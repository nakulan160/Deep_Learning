import numpy as np

x=np.array([1,0,1,1],dtype=float)

w1=np.random.randn(4,3)*0.5
b1=np.zeros(3)

w2=np.random.randn(3,2)*0.5
b2=np.zeros(2)

w3=np.random.randn(2,1)*0.5
b3=0.0

def sig(x):
    return 1/(1+np.exp(-x))

z1=np.dot(x,w1)+b1
a1=sig(z1)

z2=np.dot(a1,w2)+b2
a2=sig(z2)

z3=np.dot(a2,w3)+b3
a3=sig(z3)

print("hidden 1:",a1)
print("hidden 2:",a2)
print("output:",a3)