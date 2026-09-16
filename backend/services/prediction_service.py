import os
import torch
import torch.nn.functional as F
from PIL import Image
from backend.models.classifier import ResNetClassifier
from ml.datasets.preprocessing import get_preprocessing_transform

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class PredictionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        print("Initializing PredictionService...")
        self.device = get_device()
        self.model = ResNetClassifier(num_classes=2).to(self.device)
        
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights', 'classifier.pth')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("Loaded classifier weights.")
        else:
            print(f"[WARNING] Classifier weights not found at {weights_path}. Model will use random weights.")
            
        self.model.eval()
        self.transform = get_preprocessing_transform(train=False)
        self.classes = ['NORMAL', 'PNEUMONIA']
        
    def predict(self, image: Image.Image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        return {
            "prediction": self.classes[pred_idx],
            "confidence": float(confidence)
        }
