### credit: ChatGPT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------
# 1. Data preparation
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --------------------
# 2. Define CNN
# --------------------
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Conv layer: in_channels=1 (grayscale), out_channels=32, kernel=3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)  # After 2 pools, image is 7x7
        self.fc2   = nn.Linear(128, 10)          # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [batch,32,14,14]
        x = self.pool(F.relu(self.conv2(x)))   # [batch,64,7,7]
        x = x.view(-1, 64 * 7 * 7)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------
# 3. Training setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --------------------
# 4. Training loop
# --------------------
for epoch in range(3):  # 3 epochs for demo
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# --------------------
# 5. Evaluation
# --------------------
model.eval()
correct = 0
test_loss = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
accuracy = 100. * correct / len(test_loader.dataset)

print(f"\nTest set: Average loss {test_loss:.4f}, Accuracy {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
