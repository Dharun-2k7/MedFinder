import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import RetrievalService
from ml.datasets.medmnist_dataset import MedMNISTDataset

def run_test():
    print("========================================")
    print("MEDFINDER AI - Retrieval Test")
    print("========================================")
    
    print("\n1. Initializing Services...")
    embed_service = EmbeddingService()
    retrieval_service = RetrievalService()
    
    print("\n2. Loading Test Image...")
    # Load the test split
    test_ds = MedMNISTDataset(split='test', transform=None)
    
    # Pick the 10th image in the test set as our query
    query_idx = 10
    query_img, query_label = test_ds.dataset[query_idx]
    
    # Print ground truth
    ground_truth = "Pneumonia" if query_label[0] == 1 else "Normal"
    print(f"Query Image Ground Truth: {ground_truth}")
    
    print("\n3. Extracting Query Embedding...")
    query_embedding = embed_service.embed_image(query_img)
    print(f"Query Embedding Shape: {query_embedding.shape}")
    
    print("\n4. Searching the Gallery...")
    top_k_results = retrieval_service.search(query_embedding, top_k=5)
    
    print("\n========================================")
    print("TOP-5 SIMILAR IMAGES")
    print("========================================")
    
    for i, res in enumerate(top_k_results):
        print(f"Rank {i+1}:")
        print(f"  Similarity : {res['similarity_score']:.4f}")
        print(f"  Finding    : {res['finding']}")
        print(f"  Split      : {res['split']} (Index: {res['original_index']})")
        print("  - - - - - - - - - - - - - - - - - - -")
        
    print("\nTest Complete!")

if __name__ == "__main__":
    run_test()
