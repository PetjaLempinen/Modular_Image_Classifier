import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import cifar10cnn
from dataset import get_cifar10_loaders

def train_model(epochs=20, batch_size=64, learning_rate=0.001):
    # Set up device for GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data using dataset module
    print("Loading CIFAR-10 dataset")
    train_loader, test_loader = get_cifar10_loaders(data_dir='./data', batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = cifar10cnn(num_classes=100).to(device)
    
    # CrossEntropyLoss handles multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training Loop
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Optimize: update weights
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

    # Save the trained model weights
    os.makedirs('saved_models', exist_ok=True)
    checkpoint_path = 'saved_models/cifar10_cnn.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}.")

if __name__ == '__main__':
    train_model()