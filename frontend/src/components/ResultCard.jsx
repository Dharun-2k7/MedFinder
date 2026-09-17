import React from 'react';
import { Activity, CheckCircle, AlertTriangle } from 'lucide-react';

export default function ResultCard({ result }) {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const imageUrl = `${API_URL}/api/image/${result.split}/${result.original_index}`;
  
  const isNormal = result.finding === 'Normal';

  return (
    <div className="result-card glass-panel animate-fade-in">
      <div className="similarity-badge">
        Similarity: {result.similarity_score.toFixed(3)}
      </div>
      
      <div className="card-image-wrapper">
        <img src={imageUrl} alt="X-Ray" className="card-image" />
      </div>
      
      <div className="card-content">
        <h4 className={`card-title ${isNormal ? 'finding-normal' : 'finding-pneumonia'}`}>
          {isNormal ? <CheckCircle size={18} /> : <AlertTriangle size={18} />}
          {result.finding}
        </h4>
        
        <div className="card-meta">
          <span>{result.split} set</span>
          <span>Index: {result.original_index}</span>
        </div>
      </div>
    </div>
  );
}
