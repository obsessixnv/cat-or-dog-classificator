import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image

# Define image dimensions
IMG_WIDTH, IMG_HEIGHT = 150, 150

# Data transformations
transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = ImageFolder('data/train', transform=transform)
validation_dataset = ImageFolder('data/validation', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


model = CNN()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # Loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
#
# print('Finished Training')
#
# # Save the model
# model_save_path = 'model.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')
#
# # Evaluating the model
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in validation_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         predicted = (outputs > 0.5).float()
#         total += labels.size(0)
#         correct += (predicted == labels.view(-1, 1)).sum().item()
#
# print(f'Validation Accuracy: {100 * correct / total:.2f}%')


# Function to preprocess a single image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Function to predict the class of a single image
def predict_image(model, image_path):
    model.eval()  # Set the model to evaluation mode
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float().item()

    return 'dog' if prediction > 0.5 else 'cat'


# Load the model
model_load_path = 'model.pth'
model = CNN()  # Create a new instance of the model
model.load_state_dict(torch.load(model_load_path))
model.to(device)
model.eval()  # Set the model to evaluation mode
print(f'Model loaded from {model_load_path}')

# Specify the path to the image you want to test
test_image_path = 'predict_image_example/dog1.jpg'

# Get the prediction
prediction = predict_image(model, test_image_path)

# Print the result
print(f'The image is a {prediction}.')
