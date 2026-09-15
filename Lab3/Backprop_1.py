import numpy as np

x=np.array([1,0,1,1],dtype=float)

w=np.random.randn(4)*0.5
b=0.0

z=np.dot(x,w)+b
a=1/(1+np.exp(-z))

print("z:",z)
print("a:",a)

# target
y=1

# backprop
dz=(a-y)*a*(1-a)
dw=x*dz
db=dz

print("dw:",dw)
print("db:",db)
