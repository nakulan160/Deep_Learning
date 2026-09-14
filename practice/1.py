import torch
import torch.nn as nn


# ---------------------------------------------------------
# Single layer network
# ---------------------------------------------------------
model = nn.Linear(
    in_features=3,
    out_features=1
)


# ---------------------------------------------------------
# Example input
# ---------------------------------------------------------
x = torch.tensor([
    [2.0, 3.0, 4.0]
])


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------
output = model(x)

print("Input:")
print(x)

print("\nWeight:")
print(model.weight)

print("\nBias:")
print(model.bias)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)