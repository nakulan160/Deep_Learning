import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms

# tensors
x=torch.tensor([[1,2],[3,4]],dtype=torch.float32)
print(x)
print(x.shape)

# dataset and dataloader
data=datasets.MNIST("data",train=True,download=True,transform=transforms.ToTensor())
loader=DataLoader(data,batch_size=32,shuffle=True)

x,y=next(iter(loader))
print(x.shape)
print(y.shape)

# model
model=nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

print(model)

# autograd
a=torch.tensor(2.0,requires_grad=True)
b=a**2+3*a
b.backward()
print(a.grad)

# optimization
loss_fn=nn.CrossEntropyLoss()
opt=torch.optim.SGD(model.parameters(),lr=0.01)

for x,y in loader:
    p=model(x)
    loss=loss_fn(p,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    break

print(loss.item())

# save model
torch.save(model.state_dict(),"model.pth")

# load model
model2=nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

model2.load_state_dict(torch.load("model.pth",weights_only=True))
model2.eval()

# custom dataset
class MyData(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        return self.x[i],self.y[i]

x=torch.randn(20,5)
y=torch.randint(0,2,(20,))

data=MyData(x,y)
loader=DataLoader(data,batch_size=4,shuffle=True)

x,y=next(iter(loader))
print(x.shape)
print(y.shape)

