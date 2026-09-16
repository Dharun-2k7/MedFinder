import os
import cv2
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from backend.models.classifier import ResNetClassifier
from ml.datasets.preprocessing import get_preprocessing_transform
from backend.services.prediction_service import get_device

class ExplainabilityService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExplainabilityService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        print("Initializing ExplainabilityService (Grad-CAM)...")
        self.device = get_device()
        self.model = ResNetClassifier(num_classes=2).to(self.device)
        
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights', 'classifier.pth')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        self.model.eval()
        self.transform = get_preprocessing_transform(train=False)
        self.classes = ['NORMAL', 'PNEUMONIA']
        
        # We need gradients for Grad-CAM, so we don't use torch.no_grad()
        self.target_layer = self.model.encoder.model.layer4
        
        # Hook placeholders
        self.activations = None
        self.gradients = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, image: Image.Image, target_class=None):
        # We must make sure gradients are enabled for the parameters too, but model is eval. 
        # Actually in eval mode, parameters still have requires_grad=True by default unless we set them to False
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True
        
        self.model.zero_grad()
        logits = self.model(tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Backprop to get gradients
        score = logits[0, target_class]
        score.backward()
        
        # Get activations and gradients
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients to get weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)
        
        # Resize to original image size
        cam = cv2.resize(cam, image.size) # image.size is (width, height)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay on original image
        img_np = np.array(image.convert('RGB')) / 255.0
        # Convert to BGR for cv2
        img_bgr = img_np[:, :, ::-1]
        
        overlay = heatmap * 0.4 + img_bgr * 0.6
        overlay = overlay / np.max(overlay)
        
        # Convert back to RGB uint8
        overlay_rgb = np.uint8(255 * overlay[:, :, ::-1])
        
        # Encode to base64
        is_success, buffer = cv2.imencode(".jpg", overlay_rgb[:, :, ::-1])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str, self.classes[target_class]
