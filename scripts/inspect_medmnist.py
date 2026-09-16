import os
import medmnist
from medmnist import INFO
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def inspect_medmnist():
    print("========================================")
    print("MEDFINDER AI - MedMNIST Inspector")
    print("========================================")
    
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    
    print(f"Dataset: {info['description']}")
    print(f"Task: {info['task']}")
    print(f"Labels: {info['label']}")
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # Download and load datasets
    print("\nDownloading and loading dataset (this is very small)...")
    train_dataset = DataClass(split='train', download=True, root='./data')
    val_dataset = DataClass(split='val', download=True, root='./data')
    test_dataset = DataClass(split='test', download=True, root='./data')
    
    print(f"\nDataset Sizes:")
    print(f" - Train: {len(train_dataset)}")
    print(f" - Val: {len(val_dataset)}")
    print(f" - Test: {len(test_dataset)}")
    
    # Inspect a single sample
    img, label = train_dataset[0]
    print(f"\nSample Image Type: {type(img)}")
    print(f"Sample Image Size: {img.size}")
    print(f"Sample Image Mode: {img.mode}") # Should be 'L' for grayscale
    print(f"Sample Label: {label}")
    
    # Test our pipeline's preprocessing on MedMNIST
    print("\nTesting Preprocessing Pipeline for ResNet50...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # ResNet50 expects 3 channels, 224x224
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # Convert 'L' to 'RGB'
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        tensor_img = transform(img)
        print(f"Preprocessing Success!")
        print(f"Output Tensor Shape: {tensor_img.shape}")
        if tensor_img.shape == torch.Size([3, 224, 224]):
            print("[SUCCESS] The dataset is compatible with the existing ResNet50 pipeline!")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")

if __name__ == "__main__":
    inspect_medmnist()
