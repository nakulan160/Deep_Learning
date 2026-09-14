import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Flatten image (1 x 28 x 28 -> 784)
        self.flatten = nn.Flatten()

        # Hidden Layer 1
        self.linear1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        # Hidden Layer 2
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        # Output Layer
        self.linear3 = nn.Linear(512, 10)

    # Xavier Weight Initialization
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

    # Forward Pass
    def forward(self, x):
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)

        return x


# Load FashionMNIST Dataset
def load_data():
    training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    )

    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    )

    return training_data, test_data


# Training Function
def train(mydataloader, model, loss_fn, optimizer, device, epochs):

    size = len(mydataloader.dataset)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}\n------------------------")

        model.train()

        for batch, (X, y) in enumerate(mydataloader):

            X, y = X.to(device), y.to(device)

            # Forward Pass
            pred = model(X)

            # Loss
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")


# Evaluation Function
def evaluate(mydataloader, model, loss_fn, device):

    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for X, y in mydataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"\nTest Error:")
    print(f"Accuracy: {100 * correct:.2f}%")
    print(f"Average Loss: {test_loss:.4f}\n")


def main():

    # Load Dataset
    training_data, test_data = load_data()

    batch_size = 64

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True
    )

    # Display one batch
    for X, y in train_dataloader:
        print(f"Shape of X : {X.shape}")
        print(f"Shape of y : {y.shape}")
        break

    # Device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    print(f"\nUsing {device} device\n")

    # Model
    model = NeuralNetwork().to(device)

    # Xavier Initialization
    model.initialize_weights()

    print(model)

    # Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3
    )

    # Train
    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        device=device,
        epochs=10
    )

    # Evaluate
    evaluate(
        test_dataloader,
        model,
        loss_fn,
        device
    )

    print("End")


if __name__ == "__main__":
    main()