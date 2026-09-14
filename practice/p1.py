import os
os.environ["OMP_NUM_THREADS"]="9"
os.environ["MKL_NUM_THREADS"]="9"

import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(9)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

batch_size=64
lr=0.01
num_classes=10
num_epochs=2

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
# train_dataset=Subset(train_dataset,range(10))
# test_dataset=Subset(test_dataset,range(5))
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25),
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25),
        )
        self.block3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.25),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=3*3*64,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=256,out_features=num_classes),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

def train_cnn(model,loader,criterion,optimizer,device):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for inputs,targets in loader:
        inputs,targets=inputs.to(device),targets.to(device)

        outputs=model(inputs)
        loss=criterion(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*inputs.size(0)
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()

    epoch_loss=running_loss/total
    epoch_acc=(correct/total)*100
    return epoch_loss,epoch_acc

def evaluate_cnn(model,loader,criterion,device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for inputs,targets in loader:
            inputs,targets=inputs.to(device),targets.to(device)

            outputs=model(inputs)
            loss=criterion(outputs,targets)

            running_loss+=loss.item()*inputs.size(0)
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()

    epoch_loss=running_loss/total
    epoch_acc=(correct/total)*100
    return epoch_loss,epoch_acc

def main():
    model=CNN(num_classes=num_classes).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    best_acc=0.0

    print(f"Starting training for {num_epochs} epochs on device: {device}\n"+"-"*60)

    for epoch in range(num_epochs):
        train_loss,train_acc=train_cnn(model,train_loader,criterion,optimizer,device)
        test_loss,test_acc=evaluate_cnn(model,test_loader,criterion,device)

        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model.state_dict(),"best_cnn.pth")
            print(f"Epoch {epoch+1:02d}: New best accuracy saved -> {best_acc:.2f}%")

        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} [Acc: {train_acc:.2f}%] | "
              f"Test Loss: {test_loss:.4f} [Acc: {test_acc:.2f}%]")

    print("-"*60+f"\nTraining finished! Peak Test Accuracy achieved: {best_acc:.2f}%")

    print("Reloading best model state parameters from weights checkpoint...")
    model.load_state_dict(torch.load("best_cnn.pth",map_location=device))

    final_loss,final_acc=evaluate_cnn(model,test_loader,criterion,device)
    print(f"Verified Final Model Metrics -> Loss: {final_loss:.4f} | Accuracy: {final_acc:.2f}%")

if __name__=="__main__":
    main()