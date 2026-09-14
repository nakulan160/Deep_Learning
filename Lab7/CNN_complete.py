import os
os.environ['OMP_NUM_THREADS']='9'
os.environ['MKL_NUM_THREADS']='9'

import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(9)

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=3*3*64,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=num_classes),
        )

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def tr_CNN(model,optimizer,loader,criterion,device):
    model.train()
    running_loss = 0
    total=0
    correct=0
    batch_loss=[]
    batch_accuracy=[]
    for batch,(inputs,targets) in enumerate(loader):
        inputs,targets=inputs.to(device),targets.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        accuracy = ((predicted.eq(targets).sum().item() / targets.size(0)))*100
        batch_accuracy.append(accuracy)
        batch_loss.append(loss.item())
        print(f"Train Batch{batch+1}/{len(loader)}  |  Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%")
        running_loss += loss.item()*inputs.size(0)
        total += targets.size(0)
        correct+=predicted.eq(targets).sum().item()
    epoch_loss = running_loss/total
    epoch_accuracy = (correct/total)*100
    return epoch_loss,epoch_accuracy,batch_loss,batch_accuracy

def eval_CNN(model,loader,criterion,device):
    model.eval()
    running_loss = 0
    total=0
    correct=0
    batch_loss=[]
    batch_accuracy=[]
    with torch.no_grad():
        for batch,(inputs,targets) in enumerate(loader):
            inputs,targets=inputs.to(device),targets.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,targets)
            _, predicted = outputs.max(1)
            batch_loss.append(loss.item())
            accuracy=((predicted.eq(targets).sum().item() / targets.size(0)))*100
            batch_accuracy.append(accuracy)
            print(f"Test batch;{batch+1}/{len(loader)}  |  Loss: {loss:.4f} | Acc: {accuracy:.2f}%")
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
            running_loss += loss.item()*inputs.size(0)
    epoch_loss = running_loss/total
    epoch_accuracy = (correct/total)*100
    return epoch_loss,epoch_accuracy,batch_loss,batch_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 2
    print(f"device:{device} with num_epoch:{num_epoch}")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    train_dataset=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
    test_dataset=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
    train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)
    model = CNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    bath_train_loss=[]
    bath_train_accuracy=[]
    bath_test_loss=[]
    bath_test_accuracy=[]
    epoch_train_loss=[]
    epoch_train_accuracy=[]
    epoch_test_loss=[]
    epoch_test_accuracy=[]

    for epoch in range(num_epoch):
        epoch_loss_train,epoch_acc_train,batch_loss_train,batch_accuracy_train=tr_CNN(model,optimizer,train_loader,criterion,device)
        epoch_loss_test, epoch_acc_test, batch_loss_test, batch_accuracy_test=eval_CNN(model,test_loader,criterion,device)

        epoch_train_loss.append(epoch_loss_train)
        epoch_train_accuracy.append(epoch_acc_train)
        epoch_test_loss.append(epoch_loss_test)
        epoch_test_accuracy.append(epoch_acc_test)

        bath_train_loss.extend(batch_loss_train)
        bath_train_accuracy.extend(batch_accuracy_train)
        bath_test_loss.extend(batch_loss_test)
        bath_test_accuracy.extend(batch_accuracy_test)

        print(f'Epoch:{epoch+1}/{num_epoch}  |'
              f'Train Loss:{epoch_loss_train:.4f} | Test Loss:{epoch_loss_test:.4f} |'
              f'Train Acc:{epoch_acc_train:.4f} |   Test Acc:{epoch_acc_test:.4f} |')

    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ep_range=range(1,num_epoch+1)

    ax[0,0].plot(ep_range,epoch_train_loss,label='Train epoch Loss')
    ax[0,0].plot(ep_range,epoch_test_loss,label='Test epoch loss')
    ax[0,0].set(xlabel='Epoch',ylabel='Loss')
    ax[0,0].set_xticks(ep_range)
    ax[0,0].set_title('Train vs Test Loss')
    ax[0,0].legend(loc='best')

    ax[0,1].plot(ep_range,epoch_train_accuracy,label='Train epoch Accuracy')
    ax[0,1].plot(ep_range,epoch_test_accuracy,label='Test epoch Accuracy')
    ax[0,1].set(xlabel='Epoch',ylabel='Accuracy')
    ax[0,1].set_xticks(ep_range)
    ax[0,1].set_title('Train vs Test Accuracy')
    ax[0,1].legend(loc='best')

    ax[1,0].plot(range(1,len(bath_train_loss)+1),bath_train_loss,label='Train batch Loss')
    ax[1,0].plot(range(1,len(bath_test_loss)+1),bath_test_loss,label='Test batch loss')
    ax[1,0].set(xlabel='batch',ylabel='Loss')
    ax[1,0].set_title('Train vs Test Loss')
    ax[1,0].legend(loc='best')

    ax[1,1].plot(range(1,len(bath_train_accuracy)+1),bath_train_accuracy,label='Train batch Accuracy')
    ax[1,1].plot(range(1,len(bath_test_accuracy)+1),bath_test_accuracy,label='Test batch accuracy')
    ax[1,1].set(xlabel='batch',ylabel='Accuracy')
    ax[1,1].set_title('Train vs Test Accuracy')
    ax[1,1].legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
