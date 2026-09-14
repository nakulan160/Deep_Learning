import numpy as np

np.random.seed(1)

img=np.random.randint(0,10,(32,32))
k=np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

# convolution
conv=np.zeros((30,30))

for i in range(30):
    for j in range(30):
        a=img[i:i+3,j:j+3]
        conv[i,j]=np.sum(a*k)

print("convolution:")
print(conv)

# max pooling
pool=np.zeros((15,15))

for i in range(15):
    for j in range(15):
        a=conv[i*2:i*2+2,j*2:j*2+2]
        pool[i,j]=np.max(a)

print("max pooling:")
print(pool)