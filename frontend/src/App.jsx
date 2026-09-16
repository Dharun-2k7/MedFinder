import React, { useState } from 'react';
import axios from 'axios';
import UploadArea from './components/UploadArea';
import ResultCard from './components/ResultCard';
import './App.css';
import { Activity, Zap, CheckCircle, AlertTriangle } from 'lucide-react';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  
  const [prediction, setPrediction] = useState(null);
  const [heatmap, setHeatmap] = useState(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const handleUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setResults([]);
    setStats(null);
    setPrediction(null);
    setHeatmap(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. Hit Search API
      const searchReq = axios.post(`${API_URL}/api/search`, formData, { headers: { 'Content-Type': 'multipart/form-data' }});
      
      // 2. Hit Predict API
      const predictReq = axios.post(`${API_URL}/api/predict`, formData, { headers: { 'Content-Type': 'multipart/form-data' }});
      
      // 3. Hit Explain API
      const explainReq = axios.post(`${API_URL}/api/explain`, formData, { headers: { 'Content-Type': 'multipart/form-data' }});
      
      const [searchRes, predictRes, explainRes] = await Promise.all([searchReq, predictReq, explainReq]);
      
      if (searchRes.data && searchRes.data.results) {
        setResults(searchRes.data.results);
        setStats({ time: searchRes.data.processing_time_ms });
      }
      
      if (predictRes.data) {
        setPrediction(predictRes.data);
      }
      
      if (explainRes.data) {
        setHeatmap(explainRes.data.heatmap_base64);
      }
      
    } catch (err) {
      console.error("API calls failed:", err);
      setError(err.response?.data?.detail || "Failed to process image. Ensure backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>MedFinder AI</h1>
        <p>Medical Image Similarity & Decision-Support</p>
      </header>

      <main className="main-content">
        <section className="left-panel">
          <UploadArea onUpload={handleUpload} isLoading={isLoading} />
          
          {error && (
            <div className="glass-panel" style={{ marginTop: '1rem', padding: '1rem', color: '#f87171', border: '1px solid #f87171' }}>
              {error}
            </div>
          )}
          
          {prediction && (
            <div className="glass-panel animate-fade-in" style={{ marginTop: '1.5rem', padding: '1.5rem' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Zap size={20} color="var(--primary-color)"/> AI Diagnosis
              </h3>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem', color: prediction.prediction === 'NORMAL' ? '#4ade80' : '#f87171' }}>
                  {prediction.prediction === 'NORMAL' ? <CheckCircle size={24}/> : <AlertTriangle size={24}/>}
                  {prediction.prediction}
                </div>
                <div style={{ background: 'rgba(0,0,0,0.5)', padding: '0.4rem 0.8rem', borderRadius: '1rem', fontSize: '0.9rem', border: '1px solid var(--border-light)' }}>
                  {(prediction.confidence * 100).toFixed(1)}% Confidence
                </div>
              </div>
              
              {heatmap && (
                <div style={{ marginTop: '1rem' }}>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Grad-CAM Explainability Heatmap:</p>
                  <div style={{ width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-light)' }}>
                    <img src={`data:image/jpeg;base64,${heatmap}`} alt="Grad-CAM Heatmap" style={{ width: '100%', display: 'block' }} />
                  </div>
                </div>
              )}
            </div>
          )}
          
          {stats && (
            <div className="glass-panel" style={{ marginTop: '1.5rem', padding: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={18} color="var(--primary-color)"/>
              <span style={{ color: 'var(--text-muted)' }}>Search completed in {(stats.time / 1000).toFixed(2)}s</span>
            </div>
          )}
        </section>

        <section className="right-panel">
          <div className="results-header">
            <h3>Retrieved Cases</h3>
            <span style={{ color: 'var(--text-muted)' }}>
              {results.length > 0 ? `${results.length} similar cases found` : 'Awaiting input...'}
            </span>
          </div>

          {isLoading && (
            <div className="spinner"></div>
          )}

          {!isLoading && results.length > 0 && (
            <div className="results-grid">
              {results.map((result, idx) => (
                <ResultCard key={result.id || idx} result={result} />
              ))}
            </div>
          )}
          
          {!isLoading && results.length === 0 && !error && (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Upload an X-Ray image to find similar cases in our dataset.
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
