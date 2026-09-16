import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.datasets.medmnist_dataset import MedMNISTDataset
from backend.models.classifier import ResNetClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    train_dataset = MedMNISTDataset(split='train', dataset_root=dataset_root)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = ResNetClassifier(num_classes=2).to(device)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    print("Training classification head for 1 epoch...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).squeeze().long()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
    # Save weights
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backend', 'models', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'classifier.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")

if __name__ == "__main__":
    train()
