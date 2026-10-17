# ==========================================
# 1. Importing Required Libraries
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. Data Loading and Preprocessing
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

# Downloading MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 3. Defining Fully Connected Neural Network
# ==========================================
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten image
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. Model Training
# ==========================================
num_epochs = 5
print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training Complete!\n")

# ==========================================
# 5. Evaluation & Test Accuracy
# ==========================================
model.eval()
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(images.cpu().numpy())

accuracy = 100 * correct / total
print(f"==========================================")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"==========================================\n")

# ==========================================
# 6. Display Sample Predictions
# ==========================================
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = all_images[i].squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {all_preds[i]} (True: {all_labels[i]})")
    plt.axis('off')
plt.suptitle("Sample Predictions (Digits 0-9)", fontsize=14)
plt.show()

# ==========================================
# 7. Confusion Matrix
# ==========================================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ==========================================
# 8. Incorrect Predictions Analysis (At least 5)
# ==========================================
incorrect_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred != true]

plt.figure(figsize=(12, 5))
print(f"Total Incorrect Predictions: {len(incorrect_indices)}")

for idx, err_i in enumerate(incorrect_indices[:5]):
    plt.subplot(1, 5, idx+1)
    img = all_images[err_i].squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {all_preds[err_i]}\nTrue: {all_labels[err_i]}")
    plt.axis('off')
plt.suptitle("5 Incorrect Predictions for Analysis", fontsize=14)
plt.show()
