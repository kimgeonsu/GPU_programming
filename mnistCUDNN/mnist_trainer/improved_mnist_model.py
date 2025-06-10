import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import struct

class ImprovedMNISTNet(nn.Module):
    def __init__(self):
        super(ImprovedMNISTNet, self).__init__()
        
        # First block: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Second block: 32 -> 64 channels  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Third block: 64 -> 128 channels
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling operations: 28 -> 14 -> 7 -> 3 (with padding)
        # But let's calculate more precisely: 28/2/2/2 = 3.5, so 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.dropout1(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 7x7 -> 3x3
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model
    model = ImprovedMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Training loop
    num_epochs = 20
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Test the model
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'improved_mnist_model.pth')
        
        scheduler.step()
    
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    return model

def export_weights_for_cudnn(model_path='improved_mnist_model.pth'):
    """Export trained model weights to binary files compatible with CUDNN"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedMNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create output directory
    os.makedirs('../data_improved', exist_ok=True)
    
    def save_weights(weights, filename):
        """Save weights as binary file (float32)"""
        weights = weights.detach().cpu().numpy().astype(np.float32)
        with open(f'../data_improved/{filename}', 'wb') as f:
            f.write(weights.tobytes())
        print(f"Saved {filename} with shape {weights.shape}")
    
    def save_bias(bias, filename):
        """Save bias as binary file (float32)"""
        bias = bias.detach().cpu().numpy().astype(np.float32)
        with open(f'../data_improved/{filename}', 'wb') as f:
            f.write(bias.tobytes())
        print(f"Saved {filename} with shape {bias.shape}")
    
    # Export convolutional layers
    save_weights(model.conv1.weight, 'conv1.bin')
    save_bias(model.conv1.bias, 'conv1.bias.bin')
    
    save_weights(model.conv2.weight, 'conv2.bin')
    save_bias(model.conv2.bias, 'conv2.bias.bin')
    
    save_weights(model.conv3.weight, 'conv3.bin')
    save_bias(model.conv3.bias, 'conv3.bias.bin')
    
    save_weights(model.conv4.weight, 'conv4.bin')
    save_bias(model.conv4.bias, 'conv4.bias.bin')
    
    save_weights(model.conv5.weight, 'conv5.bin')
    save_bias(model.conv5.bias, 'conv5.bias.bin')
    
    save_weights(model.conv6.weight, 'conv6.bin')
    save_bias(model.conv6.bias, 'conv6.bias.bin')
    
    # Export fully connected layers
    save_weights(model.fc1.weight, 'fc1.bin')
    save_bias(model.fc1.bias, 'fc1.bias.bin')
    
    save_weights(model.fc2.weight, 'fc2.bin')
    save_bias(model.fc2.bias, 'fc2.bias.bin')
    
    save_weights(model.fc3.weight, 'fc3.bin')
    save_bias(model.fc3.bias, 'fc3.bias.bin')
    
    # Export batch normalization parameters
    save_weights(model.bn1.weight, 'bn1_weight.bin')
    save_bias(model.bn1.bias, 'bn1_bias.bin')
    def save_tensor(tensor, filename):
        """Save tensor as binary file (float32) - handles both parameters and buffers"""
        tensor_data = tensor.detach().cpu().numpy().astype(np.float32)
        with open(f'../data_improved/{filename}', 'wb') as f:
            f.write(tensor_data.tobytes())
        print(f"Saved {filename} with shape {tensor_data.shape}")
    
    save_tensor(model.bn1.running_mean, 'bn1_mean.bin')
    save_tensor(model.bn1.running_var, 'bn1_var.bin')
    
    save_weights(model.bn2.weight, 'bn2_weight.bin')
    save_bias(model.bn2.bias, 'bn2_bias.bin')
    save_tensor(model.bn2.running_mean, 'bn2_mean.bin')
    save_tensor(model.bn2.running_var, 'bn2_var.bin')
    
    save_weights(model.bn3.weight, 'bn3_weight.bin')
    save_bias(model.bn3.bias, 'bn3_bias.bin')
    save_tensor(model.bn3.running_mean, 'bn3_mean.bin')
    save_tensor(model.bn3.running_var, 'bn3_var.bin')
    
    save_weights(model.bn4.weight, 'bn4_weight.bin')
    save_bias(model.bn4.bias, 'bn4_bias.bin')
    save_tensor(model.bn4.running_mean, 'bn4_mean.bin')
    save_tensor(model.bn4.running_var, 'bn4_var.bin')
    
    save_weights(model.bn5.weight, 'bn5_weight.bin')
    save_bias(model.bn5.bias, 'bn5_bias.bin')
    save_tensor(model.bn5.running_mean, 'bn5_mean.bin')
    save_tensor(model.bn5.running_var, 'bn5_var.bin')
    
    save_weights(model.bn6.weight, 'bn6_weight.bin')
    save_bias(model.bn6.bias, 'bn6_bias.bin')
    save_tensor(model.bn6.running_mean, 'bn6_mean.bin')
    save_tensor(model.bn6.running_var, 'bn6_var.bin')
    
    print("All weights exported successfully!")

if __name__ == "__main__":
    print("Training improved MNIST model...")
    model = train_model()
    print("Exporting weights for CUDNN...")
    export_weights_for_cudnn()
    print("Done!") 