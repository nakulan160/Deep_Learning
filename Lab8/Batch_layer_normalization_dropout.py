import torch
from torch import nn

torch.manual_seed(1)

x=torch.randn(100,20)
y=torch.randn(100,1)

# without normalization
m=nn.Sequential(
    nn.Linear(20,50),
    nn.ReLU(),
    nn.Linear(50,1)
)

opt=torch.optim.SGD(m.parameters(),lr=0.01)

for i in range(20):
    p=m(x)
    loss=((p-y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("without normalization:",loss.item())

# batch normalization
m=nn.Sequential(
    nn.Linear(20,50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    nn.Linear(50,1)
)

opt=torch.optim.SGD(m.parameters(),lr=0.01)

for i in range(20):
    p=m(x)
    loss=((p-y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("batch normalization:",loss.item())

# layer normalization
m=nn.Sequential(
    nn.Linear(20,50),
    nn.LayerNorm(50),
    nn.ReLU(),
    nn.Linear(50,1)
)

opt=torch.optim.SGD(m.parameters(),lr=0.01)

for i in range(20):
    p=m(x)
    loss=((p-y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("layer normalization:",loss.item())

# dropout
m=nn.Sequential(
    nn.Linear(20,50),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(50,1)
)

opt=torch.optim.SGD(m.parameters(),lr=0.01)

for i in range(20):
    p=m(x)
    loss=((p-y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print("dropout:",loss.item())

# SGD
w=torch.randn(5,5,requires_grad=True)
opt=torch.optim.SGD([w],lr=0.01)

loss=(w-y[:5,:5]).pow(2).mean()
loss.backward()
opt.step()

print("SGD:",loss.item())

# momentum
w=torch.randn(5,5,requires_grad=True)
opt=torch.optim.SGD([w],lr=0.01,momentum=0.9)

loss=(w-y[:5,:5]).pow(2).mean()
loss.backward()
opt.step()

print("Momentum:",loss.item())

# AdaGrad
w=torch.randn(5,5,requires_grad=True)
opt=torch.optim.Adagrad([w],lr=0.01)

loss=(w-y[:5,:5]).pow(2).mean()
loss.backward()
opt.step()

print("AdaGrad:",loss.item())