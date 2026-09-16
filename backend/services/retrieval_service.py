import os
import numpy as np
import pandas as pd

class RetrievalService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RetrievalService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        """Loads embeddings and metadata into memory."""
        print("Initializing RetrievalService...")
        
        emb_path = 'index/embeddings.npy'
        meta_path = 'index/metadata.csv'
        
        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            print("[WARNING] Index files not found! Retrieval will fail until embeddings are extracted.")
            self.embeddings = None
            self.metadata = None
            return
            
        print(f"Loading embeddings from {emb_path}...")
        self.embeddings = np.load(emb_path) # Shape: (N, 2048)
        
        print(f"Loading metadata from {meta_path}...")
        self.metadata = pd.read_csv(meta_path)
        
        # Verify sizes match
        assert len(self.embeddings) == len(self.metadata), "Embeddings and metadata size mismatch!"
        print(f"Loaded {len(self.embeddings)} indexed images.")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Searches the gallery for the top-k most similar embeddings.
        
        Args:
            query_embedding (np.ndarray): Normalized 1D array of shape (2048,).
            top_k (int): Number of results to return (max 20).
            
        Returns:
            list[dict]: Top-K results containing similarity score and metadata.
        """
        if self.embeddings is None:
            raise ValueError("Index not loaded.")
            
        top_k = min(max(1, top_k), 20)
        
        # Calculate Cosine Similarity
        # Since both gallery and query are L2-normalized, cosine similarity is just the dot product.
        # embeddings: (N, 2048), query_embedding: (2048,)
        # result: (N,)
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top-K indices (partition for speed, then sort)
        # Using argpartition is O(N) instead of O(N log N) for full sort
        if len(scores) > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            # Sort the top_k indices by score descending
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
            
        results = []
        for idx in top_indices:
            score = scores[idx]
            row = self.metadata.iloc[idx]
            
            result = {
                "similarity_score": float(score),
                "id": int(row['id']),
                "split": str(row['split']),
                "original_index": int(row['original_index']),
                "finding": str(row['finding'])
            }
            results.append(result)
            
        return results
