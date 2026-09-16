import torch
import torch.nn as nn
from backend.models.resnet_encoder import ResNetEncoder

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.encoder = ResNetEncoder()
        # Add the classification head
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.fc(features)
        return logits
