import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Check for available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the data
filename = "EISmat/xy_data_33k_5circuit_v2.mat"
x = scipy.io.loadmat(filename)["x_data"]
y = scipy.io.loadmat(filename)["y_data"]
y = np.squeeze(y)
x = np.swapaxes(x, 1, 2)
y = np.eye(5)[y]  # One-hot encode labels

# Data Augmentation
new_shape = list(x.shape)
new_shape[-1] += 3
new_x = np.zeros(new_shape)
new_x[:, :, :3] = x
new_x[:, :, 3] = x[:, :, 0] * -1
new_x[:, :, 4] = x[:, :, 1] * -1
new_x[:, :, 5] = x[:, :, 2] * -1

# Split data
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
batch_size = 256
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class ConvFeatNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvFeatNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=32, padding='same')
        self.conv2 = nn.Conv1d(64, 128, 16, padding='same')
        self.conv3 = nn.Conv1d(128, 256, 8, padding='same')
        self.conv4 = nn.Conv1d(256, 512, 4, padding='same')
        self.conv5 = nn.Conv1d(512, 768, 2, padding='same')
        
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(768 * input_shape[1] // 32, 1024)  # Adjusting for input length
        self.fc2 = nn.Linear(1024, 5)  # Output 5 classes
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool1d(x, 1)  # Global Average Pooling
        
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = ConvFeatNet(x_train.shape[1:]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 400
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%y_%m_%d") + "/Experiment"
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.argmax(dim=1))  # Target is one-hot encoded, so we use argmax
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target.argmax(dim=1)).sum().item()
    
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(100 * correct / total)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.argmax(dim=1))
            val_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.argmax(dim=1)).sum().item()

    val_losses.append(val_loss / len(test_loader))
    val_accs.append(100 * correct / total)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accs[-1]:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accs[-1]:.2f}%")

# Plotting the training and validation metrics
plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(log_dir + "/accuracy.png")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(log_dir + "/loss.png")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(target.argmax(dim=1).cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["C1", "C2", "C3", "C4", "C5"])
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap="Blues", ax=ax)
plt.title(f"Confusion Matrix\nAccuracy: {100 * (np.array(y_true) == np.array(y_pred)).mean():.2f}%")
plt.savefig(log_dir + "/confusion_matrix.png")
