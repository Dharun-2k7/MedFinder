import torch
import numpy as np
from PIL import Image

from backend.models.resnet_encoder import ResNetEncoder
from ml.datasets.preprocessing import get_preprocessing_transform

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        """Initializes the model on the best available device."""
        print("Initializing EmbeddingService...")
        self.device = get_device()
        
        self.encoder = ResNetEncoder().to(self.device)
        self.encoder.eval()
        
        # We use the inference (val) preprocessing transform
        self.transform = get_preprocessing_transform(train=False)
        
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embeds a single PIL image.
        
        Args:
            image (PIL.Image.Image): The input image.
            
        Returns:
            np.ndarray: A normalized 1D numpy array of shape (2048,).
        """
        # 1. Preprocess
        tensor = self.transform(image)
        # Add batch dimension: (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # 2. Extract features
        with torch.no_grad():
            features = self.encoder(tensor)
            
        # 3. L2 Normalize
        features_np = features.cpu().numpy()[0]
        norm = np.linalg.norm(features_np)
        if norm > 0:
            features_np = features_np / norm
            
        return features_np
