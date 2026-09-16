import React, { useCallback, useState } from 'react';
import { UploadCloud, FileImage, X } from 'lucide-react';

export default function UploadArea({ onUpload, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = (file) => {
    if (!file) return;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
    
    // Pass to parent
    onUpload(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const clearPreview = (e) => {
    e.stopPropagation();
    setPreview(null);
  };

  return (
    <div className="glass-panel" style={{ padding: '2rem' }}>
      <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <FileImage size={24} color="var(--primary-color)"/> 
        Query Image
      </h3>
      
      {preview ? (
        <div className="preview-container animate-fade-in">
          <img src={preview} alt="Preview" className="preview-image" />
          <button className="clear-btn" onClick={clearPreview} disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Clear Selection'}
          </button>
        </div>
      ) : (
        <label 
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <UploadCloud size={64} className="upload-icon" />
          <div className="upload-text">Drag & Drop X-Ray Image</div>
          <div className="upload-subtext">or click to browse files (JPG, PNG)</div>
          <input 
            type="file" 
            accept="image/jpeg, image/png" 
            onChange={handleChange} 
            style={{ display: 'none' }}
          />
        </label>
      )}
    </div>
  );
}
