import numpy as np
import matplotlib.pyplot as plt
from cryptography.x509 import SignatureAlgorithmOID

#Functions
z=np.linspace(-10,10,100)
sigmoid=1/(1+np.exp(-z))
tanh=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
relu=np.maximum(0,z)
leaky_relu=np.where(z>0,z,z*0.01)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(z,sigmoid)
plt.xlabel('z')
plt.ylabel('sigmoid values')
plt.title('sigmoid')

plt.subplot(2,2,2)
plt.plot(z,tanh)
plt.xlabel('z')
plt.ylabel('tanh values')
plt.title('tanh')

plt.subplot(2,2,3)
plt.plot(z,relu)
plt.xlabel('z')
plt.ylabel('relu values')
plt.title('relu')

plt.subplot(2,2,4)
plt.plot(z,leaky_relu)
plt.xlabel('z')
plt.ylabel('leaky relu values')
plt.title('leaky_relu')

plt.tight_layout()
plt.show()


#Derivatives
sigmoid_derivative=sigmoid*(1-sigmoid)
tanh_derivative=1-tanh**2
relu_derivative=np.where(z<0,0,1)
leaky_relu_derivative=np.where(z>0,1,0.01)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(z,sigmoid_derivative)
plt.xlabel('z')
plt.ylabel('sigmoid derivative values')
plt.title('sigmoid_derivative')

plt.subplot(2,2,2)
plt.plot(z,tanh_derivative)
plt.xlabel('z')
plt.ylabel('tanh derivative values')
plt.title('tanh_derivative')

plt.subplot(2,2,3)
plt.plot(z,relu_derivative)
plt.xlabel('z')
plt.ylabel('relu derivative values')
plt.title('relu_derivative')

plt.subplot(2,2,4)
plt.plot(z,leaky_relu_derivative)
plt.xlabel('z')
plt.ylabel('leaky relu derivative values')
plt.title('leaky_relu_derivative')

plt.tight_layout()
plt.show()

#softmax
softmax=np.exp(z)/np.sum(np.exp(z))
softmax_derivative=softmax*(1-softmax)
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(z,softmax)
plt.xlabel('z')
plt.ylabel('softmax values')
plt.title('softmax')
plt.subplot(1,2,2)
plt.plot(z,softmax_derivative)
plt.xlabel('z')
plt.ylabel('softmax_derivative values')
plt.title('softmax_derivative')
plt.tight_layout()
plt.show()

#function    min     max
#sigmoid      0       1
#tanh        -1       1
#relu         0       inf
#leaky_relu  -inf     inf

#only tanh is 0 centred

# gradient value is    inf     -inf
# sigmoid              0         0
# tanh                 0         0
# relu                 1         0
# leaky_relu           1         alpha

#tanh(z)=2sigmoid(2z)-1
