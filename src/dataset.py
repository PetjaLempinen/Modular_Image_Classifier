import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# prepares the training and testing data for cifar-100
def get_cifar100_loaders(data_dir='./data', batch_size=64):
    
    # Define data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    # Define transforms for testing 
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    # Download and load the training dataset
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )

    # Download and load the testing dataset
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, test_loader