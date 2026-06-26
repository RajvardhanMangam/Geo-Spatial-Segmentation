/**
 * UploadPanel — Mission upload section.
 *
 * All upload + inference-start logic is identical to the original.
 * Only the visual presentation changes.
 */
import React, { useCallback, useState, useMemo } from 'react';
import { uploadTif, startInference } from '../services/api';
import { useStore } from '../store/useStore';

function computeAreaKm2(metadata) {
  if (!metadata?.bounds) return null;
  // Coordinates live on metadata.bounds; CRS lives directly on metadata
  const { left, right, top, bottom } = metadata.bounds;
  const crs = metadata.crs || '';
  const width  = Math.abs(right - left);
  const height = Math.abs(top - bottom);
  const crsStr = crs.toUpperCase();
  if (crsStr.includes('4326')) {
    const latMid = (top + bottom) / 2;
    const mPerDegLon = 111320 * Math.cos((latMid * Math.PI) / 180);
    const mPerDegLat = 110574;
    return ((width * mPerDegLon * height * mPerDegLat) / 1e6).toFixed(2);
  }
  // UTM and other projected CRS: bounds already in metres
  return ((width * height) / 1e6).toFixed(2);
}

export default function UploadPanel() {
  const [dragOver, setDragOver] = useState(false);

  const {
    uploadStatus, uploadProgress, imageMetadata,
    confidenceThreshold,
    setUploadId, setUploadProgress, setUploadStatus, setImageMetadata,
    setJobId, setJobStatus, setConfidenceThreshold,
    reset,
  } = useStore();

  const areaKm2 = useMemo(() => computeAreaKm2(imageMetadata), [imageMetadata]);

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
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const onInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div className="section upload-panel">
      <div className="section-header">
        <span className="section-title">Upload</span>
        {uploadStatus !== 'idle' && (
          <span
            className={`status-badge status-badge--${
              uploadStatus === 'complete' ? 'completed' : uploadStatus === 'error' ? 'failed' : 'running'
            }`}
          >
            {uploadStatus === 'complete' ? 'Ready' : uploadStatus === 'error' ? 'Error' : 'Uploading'}
          </span>
        )}
      </div>

      {/* Idle: drop zone */}
      {uploadStatus === 'idle' && (
        <>
          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <div className="drop-icon-wrap">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div className="drop-title">Drop GeoTIFF here</div>
            <div className="drop-sub">or click to browse files</div>
            <div className="drop-hint">GeoTIFF · EPSG:32644 / 3857 · up to 6 GB</div>
            <input
              id="file-input"
              type="file"
              accept=".tif,.tiff"
              style={{ display: 'none' }}
              onChange={onInputChange}
            />
          </div>

          <div className="threshold-row">
            <span className="threshold-label">Min confidence</span>
            <input
              type="range"
              min="0.05"
              max="0.5"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            />
            <span className="threshold-value">{Math.round(confidenceThreshold * 100)}%</span>
          </div>
        </>
      )}

      {/* Uploading: progress bar */}
      {uploadStatus === 'uploading' && (
        <div className="upload-progress-wrap animate-in">
          <div className="upload-progress-label">
            <span>Uploading GeoTIFF...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill progress-fill--upload"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Complete: file metadata */}
      {uploadStatus === 'complete' && imageMetadata && (
        <div className="meta-grid animate-in">
          <div className="meta-row">
            <span className="meta-label">File</span>
            <span className="meta-value">{imageMetadata.filename || 'orthophoto.tif'}</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">Resolution</span>
            <span className="meta-value">{imageMetadata.width} × {imageMetadata.height} px</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">Bands</span>
            <span className="meta-value">{imageMetadata.bands}</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">CRS</span>
            <span className="meta-value">{imageMetadata.crs}</span>
          </div>
          {areaKm2 && (
            <div className="meta-row">
              <span className="meta-label">Area</span>
              <span className="meta-value">{areaKm2} km²</span>
            </div>
          )}
          <div className="meta-row">
            <span className="meta-label">Chunks</span>
            <span className="meta-value">{imageMetadata.total_inference_chunks}</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {uploadStatus === 'error' && (
        <div className="error-box animate-in">
          <span>Upload failed</span>
          <button className="retry-btn" onClick={reset}>Retry</button>
        </div>
      )}
    </div>
  );
}
