import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        
        # First convolutional block: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # padding=1 to maintain size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Second convolutional block: 32 -> 64 channels  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # padding=1 to maintain size
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        # After 2 pooling operations: 28 -> 14 -> 7, so 64 * 7 * 7 = 3136
        self.fc = nn.Linear(64 * 7 * 7, 10)
        
    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))  # Apply ReLU activation
        x = self.pool1(x)  # 28x28 -> 14x14
        
        # Second convolutional block
        x = F.relu(self.conv2(x))  # Apply ReLU activation
        x = self.pool2(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64*7*7)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)  # No activation here, will use CrossEntropyLoss
        
        return x

def train_simple_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations - same as original for fair comparison
    train_transform = transforms.Compose([
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
    model = SimpleMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training loop
    num_epochs = 15  # Fewer epochs for simple model
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
            
            if batch_idx % 200 == 0:
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
            torch.save(model.state_dict(), 'simple_mnist_model.pth')
        
        scheduler.step()
    
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    return model

def export_simple_weights_for_cudnn(model_path='simple_mnist_model.pth'):
    """Export trained simple model weights to binary files compatible with CUDNN"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create output directory
    os.makedirs('../data_simple', exist_ok=True)
    
    def save_weights(weights, filename):
        """Save weights as binary file (float32)"""
        weights = weights.detach().cpu().numpy().astype(np.float32)
        with open(f'../data_simple/{filename}', 'wb') as f:
            f.write(weights.tobytes())
        print(f"Saved {filename} with shape {weights.shape}")
    
    def save_bias(bias, filename):
        """Save bias as binary file (float32)"""
        bias = bias.detach().cpu().numpy().astype(np.float32)
        with open(f'../data_simple/{filename}', 'wb') as f:
            f.write(bias.tobytes())
        print(f"Saved {filename} with shape {bias.shape}")
    
    # Export convolutional layers (only 2 layers)
    save_weights(model.conv1.weight, 'conv1.bin')
    save_bias(model.conv1.bias, 'conv1.bias.bin')
    
    save_weights(model.conv2.weight, 'conv2.bin')
    save_bias(model.conv2.bias, 'conv2.bias.bin')
    
    # Export fully connected layer (only 1 layer)
    save_weights(model.fc.weight, 'fc1.bin')
    save_bias(model.fc.bias, 'fc1.bias.bin')
    
    print("Simple model weights exported successfully!")
    print("Architecture: Conv(1->32) -> Pool -> Conv(32->64) -> Pool -> FC(3136->10)")

if __name__ == "__main__":
    print("Training simple MNIST model...")
    print("Architecture: Conv(1->32) -> Pool -> Conv(32->64) -> Pool -> Dropout -> FC(3136->10)")
    model = train_simple_model()
    print("Exporting weights for CUDNN...")
    export_simple_weights_for_cudnn()
    print("Done!") 