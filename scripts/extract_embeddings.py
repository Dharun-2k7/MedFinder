import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root to the path so we can import from backend and ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.datasets.medmnist_dataset import MedMNISTDataset
from backend.models.resnet_encoder import ResNetEncoder

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    return embeddings / norms

def extract_embeddings():
    print("========================================")
    print("MEDFINDER AI - Embedding Extraction")
    print("========================================")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize the encoder
    print("Loading ResNet50 encoder...")
    encoder = ResNetEncoder().to(device)
    encoder.eval()
    
    # Ensure index directory exists
    os.makedirs('index', exist_ok=True)
    
    all_embeddings = []
    metadata = []
    
    global_index = 0
    
    # Process both train and val splits to build the gallery
    splits = ['train', 'val', 'test']
    batch_size = 64
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        try:
            # For extraction, we do not want data augmentation, so we treat all splits as 'val' for transforms
            dataset = MedMNISTDataset(split=split)
            # Override transform to use the validation (non-augmented) transform
            from ml.datasets.preprocessing import get_preprocessing_transform
            dataset.transform = get_preprocessing_transform(train=False)
            
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Extracting {split}")):
                    images = images.to(device)
                    
                    # Forward pass
                    features = encoder(images)
                    
                    # Convert to numpy and store
                    features_np = features.cpu().numpy()
                    all_embeddings.append(features_np)
                    
                    # Track metadata
                    for i in range(len(labels)):
                        metadata.append({
                            'id': global_index,
                            'split': split,
                            'original_index': batch_idx * batch_size + i,
                            'label': int(labels[i].item()),
                            'finding': 'Pneumonia' if int(labels[i].item()) == 1 else 'Normal'
                        })
                        global_index += 1
        except Exception as e:
            print(f"[ERROR] Failed processing {split} split: {e}")
            
    print("\nAggregation and Normalization...")
    final_embeddings = np.vstack(all_embeddings)
    print(f"Raw embeddings shape: {final_embeddings.shape}")
    
    # Normalize for cosine similarity
    normalized_embeddings = normalize_embeddings(final_embeddings)
    
    print("Saving artifacts...")
    # Save embeddings
    np.save('index/embeddings.npy', normalized_embeddings)
    print("Saved index/embeddings.npy")
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv('index/metadata.csv', index=False)
    print("Saved index/metadata.csv")
    
    print("\nExtraction Complete!")

if __name__ == "__main__":
    extract_embeddings()
