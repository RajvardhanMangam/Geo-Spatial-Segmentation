import React, { useCallback, useState } from 'react';
import { uploadTif, startInference } from '../services/api';
import { useStore } from '../store/useStore';

export default function UploadPanel() {
  const [dragOver, setDragOver] = useState(false);
  const {
    uploadStatus, uploadProgress, imageMetadata,
    confidenceThreshold,
    setUploadId, setUploadProgress, setUploadStatus, setImageMetadata,
    setJobId, setJobStatus, setConfidenceThreshold,
    reset,
  } = useStore();

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    if (!file.name.match(/\.(tif|tiff)$/i)) {
      alert('Please upload a GeoTIFF (.tif or .tiff) file.');
      return;
    }
    reset();
    setUploadStatus('uploading');
    try {
      const { uploadId, metadata } = await uploadTif(file, (p) => setUploadProgress(p));
      setUploadId(uploadId);
      setImageMetadata(metadata);
      setUploadStatus('complete');
      const { job_id } = await startInference(uploadId, confidenceThreshold);
      setJobId(job_id);
      setJobStatus('queued');
    } catch (err) {
      console.error('Upload error:', err);
      setUploadStatus('error');
      alert(`Error: ${err.response?.data?.detail || err.message}`);
    }
  }, [confidenceThreshold]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  return (
    <div className="panel">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-icon cyan-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <div className="panel-label">
          <div className="panel-title">Upload Orthophoto</div>
          <div className="panel-subtitle">GeoTIFF · up to 6 GB</div>
        </div>
      </div>

      {/* Idle: drop zone + threshold */}
      {uploadStatus === 'idle' && (
        <>
          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <div className="drop-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 2L12 16M12 2L8 6M12 2L16 6"/>
                <path d="M3 18a9 9 0 0 0 18 0"/>
              </svg>
            </div>
            <div className="drop-text">Drop .tif / .tiff here</div>
            <div className="drop-sub">or click to browse</div>
            <div className="drop-badge">.tif · .tiff · EPSG:32xxx · 3857</div>
            <input
              id="file-input"
              type="file"
              accept=".tif,.tiff"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>

          <div className="threshold-section">
            <div className="threshold-label-row">
              <span className="threshold-label">Min Confidence</span>
              <span className="threshold-value">{Math.round(confidenceThreshold * 100)}%</span>
            </div>
            <input
              type="range"
              className="slider"
              min="0.05" max="0.5" step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            />
          </div>
        </>
      )}

      {/* Uploading */}
      {uploadStatus === 'uploading' && (
        <div>
          <div className="progress-label-row">
            <span className="progress-label-text">Uploading file…</span>
            <span className="progress-label-pct">{uploadProgress}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill fill-upload" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {/* Complete: show metadata */}
      {uploadStatus === 'complete' && imageMetadata && (
        <div className="meta-grid">
          <div className="meta-card">
            <div className="meta-card-label">Dimensions</div>
            <div className="meta-card-value">
              {imageMetadata.width?.toLocaleString()} × {imageMetadata.height?.toLocaleString()}
            </div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Bands</div>
            <div className="meta-card-value">{imageMetadata.bands}</div>
          </div>
          <div className="meta-card span-2">
            <div className="meta-card-label">CRS</div>
            <div className="meta-card-value">{imageMetadata.crs}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Chunks</div>
            <div className="meta-card-value">{imageMetadata.total_inference_chunks}</div>
          </div>
          <div className="meta-card">
            <div className="meta-card-label">Confidence</div>
            <div className="meta-card-value">{Math.round(confidenceThreshold * 100)}%</div>
          </div>
        </div>
      )}

      {/* Error */}
      {uploadStatus === 'error' && (
        <div className="error-box">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          Upload failed.
          <button className="btn-retry" onClick={reset}>Retry</button>
        </div>
      )}
    </div>
  );
}
