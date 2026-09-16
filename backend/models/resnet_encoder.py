import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # Load pre-trained ResNet50 using the latest weights standard
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the classification head to output the 2048-dimensional features
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Feature embeddings of shape (B, 2048)
        """
        features = self.model(x)
        return features
        
    def extract_features(self, x):
        """
        Alias for forward pass to explicitly indicate feature extraction.
        """
        return self(x)
