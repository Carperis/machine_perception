import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

# Download MNIST
torchvision.datasets.MNIST(".", download=True)
device = torch.device("mps")


# Create your network here (do not change this name)
class DigitClassification(torch.nn.Module):
    def __init__(self):
        super(DigitClassification, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Instantiate your network here
model = DigitClassification().to(device)
print(model)

# Where your trained model will be saved (and where the autograder will load it)
model_path = "model.pth"

# Train your network here
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root=".", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    print("Epoch %d/%d" % (epoch + 1, num_epochs))

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # save model for each epoch
    model_path_temp = f"./model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path_temp)

torch.save(model.state_dict(), model_path)

epochs = list(range(1, num_epochs + 1))

fig, ax1 = plt.subplots()
color = "black"

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss", color=color)
ax1.plot(epochs, train_losses, label="Train Loss", color="tab:red", linestyle="dashed")
ax1.plot(epochs, test_losses, label="Test Loss", color="tab:blue", linestyle="dashed")
ax1.tick_params(axis="y", labelcolor=color)
plt.legend()

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color=color)  # we already handled the x-label with ax1
ax2.plot(epochs, train_accuracies, label="Train Accuracy", color="tab:red")
ax2.plot(epochs, test_accuracies, label="Test Accuracy", color="tab:blue")
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.title("Loss and Accuracy over Epochs")
plt.legend()
plt.show()
