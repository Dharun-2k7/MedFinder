import torch
from torchvision import transforms
from PIL import Image

def get_preprocessing_transform(train=False):
    """
    Builds the preprocessing pipeline for ResNet50 embedding extraction on MedMNIST.
    The source images are 28x28 grayscale. We convert them to 3 channels and resize to 224x224.
    
    Args:
        train (bool): Whether to include data augmentation.
        
    Returns:
        torchvision.transforms.Compose
    """
    # Standard ImageNet normalization for ResNet50
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if train:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

def load_image(image_path):
    """
    Loads an image from path. Not strictly used by MedMNIST wrapper, but kept for compatibility.
    """
    img = Image.open(image_path)
    return img.convert('RGB')
