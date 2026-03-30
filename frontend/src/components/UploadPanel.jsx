/**
 * UploadPanel — drag-and-drop GeoTIFF upload with chunked progress.
 */
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
      const { uploadId, metadata } = await uploadTif(file, (p) => {
        setUploadProgress(p);
      });
      setUploadId(uploadId);
      setImageMetadata(metadata);
      setUploadStatus('complete');

      // Auto-start inference
      const { job_id } = await startInference(uploadId, confidenceThreshold);
      setJobId(job_id);
      setJobStatus('queued');
    } catch (err) {
      console.error('Upload/inference error:', err);
      setUploadStatus('error');
      alert(`Error: ${err.response?.data?.detail || err.message}`);
    }
  }, [confidenceThreshold]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div className="upload-panel">
      <div className="panel-header">
        <span className="panel-title">UPLOAD ORTHOPHOTO</span>
        <span className="panel-subtitle">GeoTIFF · up to 6 GB</span>
      </div>

      {uploadStatus === 'idle' && (
        <>
          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <div className="drop-icon">⊕</div>
            <div className="drop-text">Drop .tif file here</div>
            <div className="drop-sub">or click to browse</div>
            <input
              id="file-input"
              type="file"
              accept=".tif,.tiff"
              style={{ display: 'none' }}
              onChange={onInputChange}
            />
          </div>

          <div className="threshold-row">
            <label>Min confidence</label>
            <input
              type="range" min="0.05" max="0.5" step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            />
            <span>{Math.round(confidenceThreshold * 100)}%</span>
          </div>
        </>
      )}

      {uploadStatus === 'uploading' && (
        <div className="upload-progress">
          <div className="progress-label">
            Uploading… {uploadProgress}%
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill upload-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {uploadStatus === 'complete' && imageMetadata && (
        <div className="meta-grid">
          <MetaRow label="Size" value={`${imageMetadata.width} × ${imageMetadata.height} px`} />
          <MetaRow label="Bands" value={imageMetadata.bands} />
          <MetaRow label="CRS" value={imageMetadata.crs} />
          <MetaRow label="Chunks" value={imageMetadata.total_inference_chunks} />
        </div>
      )}

      {uploadStatus === 'error' && (
        <div className="error-box">
          Upload failed.{' '}
          <button className="retry-btn" onClick={reset}>Retry</button>
        </div>
      )}
    </div>
  );
}

function MetaRow({ label, value }) {
  return (
    <div className="meta-row">
      <span className="meta-label">{label}</span>
      <span className="meta-value">{value}</span>
    </div>
  );
}
