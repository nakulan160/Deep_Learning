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

# forward pass
z1=np.dot(x,w1)+b1
a1=sig(z1)

z2=np.dot(a1,w2)+b2
a2=sig(z2)

z3=np.dot(a2,w3)+b3
a3=sig(z3)

print("hidden 1:",a1)
print("hidden 2:",a2)
print("output:",a3)

# target
y=1

# output error
dz3=(a3-y)*a3*(1-a3)
dw3=np.outer(a2,dz3)
db3=dz3

# hidden layer 2
dz2=np.dot(dz3,w3.T)*a2*(1-a2)
dw2=np.outer(a1,dz2)
db2=dz2

# hidden layer 1
dz1=np.dot(dz2,w2.T)*a1*(1-a1)
dw1=np.outer(x,dz1)
db1=dz1

print("\ndw3:",dw3)
print("db3:",db3)

print("\ndw2:",dw2)
print("db2:",db2)

print("\ndw1:",dw1)
print("db1:",db1)
