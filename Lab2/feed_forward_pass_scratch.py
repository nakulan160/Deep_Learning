import numpy as np

def calc_z(x,w,bias):
    return np.matmul(x,w)+bias
def activation(z):
    return np.maximum(0,z)

x=np.array([[0.3,-1.2]])
w1=np.array([
    [0.1,-0.1,0.2],
    [-1.1,0.4,1.1]
])

bias=np.array([3,4,6])
z1=calc_z(x,w1,bias)
print("z1:",z1)
a1=activation(z1)
print("a1:",a1)
w2=np.array([
    [0.2,0.3],
    [0.1,-0.1],
    [-0.2,-0.1]
])
bias=np.array([3,4])
z2=calc_z(a1,w2,bias)
print("z2:",z2)
a2=activation(z2)
print("a2:",a2)
w3=np.array([[0.3],[-0.3]])
bias=np.array([3])
z3=calc_z(a2,w3,bias)
print("z3:",z3)
a3=activation(z3)
print("a3:",a3)