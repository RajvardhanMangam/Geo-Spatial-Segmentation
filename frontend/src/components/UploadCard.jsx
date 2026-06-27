import React, { useCallback, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { uploadTif } from '../services/api';
import { useStore } from '../store/useStore';

export default function UploadCard() {
  const [dragOver, setDragOver] = useState(false);
  const [speed, setSpeed]       = useState(0);
  const [chunksDone, setChunksDone] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const startTimeRef = useRef(null);

  const {
    uploadStatus, uploadProgress, imageMetadata,
    setUploadId, setUploadProgress, setUploadStatus, setImageMetadata,
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
    startTimeRef.current = Date.now();

    const CHUNK_BYTES = 8 * 1024 * 1024;
    const total = Math.ceil(file.size / CHUNK_BYTES);
    setTotalChunks(total);

    try {
      const { uploadId, metadata } = await uploadTif(file, (p, chunkIdx) => {
        setUploadProgress(p);
        if (chunkIdx !== undefined) setChunksDone(chunkIdx + 1);
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        if (elapsed > 0 && p > 0) {
          const bytesUploaded = (p / 100) * file.size;
          setSpeed(bytesUploaded / elapsed);
        }
      });
      setUploadId(uploadId);
      setImageMetadata(metadata);
      setUploadStatus('complete');
    } catch (err) {
      console.error('Upload error:', err);
      setUploadStatus('error');
      alert(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const elapsed = startTimeRef.current
    ? Math.floor((Date.now() - startTimeRef.current) / 1000)
    : 0;
  const remaining = uploadProgress > 0
    ? Math.round((elapsed / uploadProgress) * (100 - uploadProgress))
    : null;

  const formatBytes = (b) => {
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB/s';
    if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB/s';
    return (b / 1e3).toFixed(0) + ' KB/s';
  };

  const formatTime = (s) => {
    if (s === null || s < 0) return '—';
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}m ${s % 60}s`;
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon card-icon-blue">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <div>
          <div className="card-title">Upload Orthophoto</div>
          <div className="card-subtitle">GeoTIFF · up to 6 GB</div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {/* ── Idle: drop zone ── */}
        {uploadStatus === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
          >
            <div
              className={`drop-zone${dragOver ? ' drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => document.getElementById('geotiff-file-input').click()}
            >
              <div className="drop-icon-wrap">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <polyline points="16 16 12 12 8 16" />
                  <line x1="12" y1="12" x2="12" y2="21" />
                  <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
                </svg>
              </div>
              <div className="drop-title">Drop GeoTIFF here</div>
              <div className="drop-sub">or click to browse files</div>
              <div className="drop-formats">
                <span className="drop-format-tag">.tif</span>
                <span className="drop-format-tag">.tiff</span>
                <span className="drop-format-tag">GeoTIFF</span>
                <span className="drop-format-tag">6 GB max</span>
              </div>
              <input
                id="geotiff-file-input"
                type="file"
                accept=".tif,.tiff"
                style={{ display: 'none' }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          </motion.div>
        )}

        {/* ── Uploading ── */}
        {uploadStatus === 'uploading' && (
          <motion.div
            key="uploading"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="upload-progress-info">
              <div className="upload-info-item">
                <div className="upload-info-label">Speed</div>
                <div className="upload-info-value">{speed > 0 ? formatBytes(speed) : '—'}</div>
              </div>
              <div className="upload-info-item">
                <div className="upload-info-label">Remaining</div>
                <div className="upload-info-value">{formatTime(remaining)}</div>
              </div>
              <div className="upload-info-item">
                <div className="upload-info-label">Chunks</div>
                <div className="upload-info-value">{chunksDone}/{totalChunks}</div>
              </div>
              <div className="upload-info-item">
                <div className="upload-info-label">Progress</div>
                <div className="upload-info-value">{uploadProgress}%</div>
              </div>
            </div>
            <div className="progress-bar-wrap">
              <div className="progress-bar-header">
                <span className="progress-bar-label">Uploading…</span>
                <span className="progress-bar-pct">{uploadProgress}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill fill-upload" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          </motion.div>
        )}

        {/* ── Complete: success + metadata ── */}
        {uploadStatus === 'complete' && imageMetadata && (
          <motion.div
            key="complete"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
          >
            <div className="upload-success-banner">
              <div className="upload-success-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
              <div>
                <div className="upload-success-text">Upload Successful</div>
                <div className="upload-success-filename">
                  {imageMetadata.crs}
                </div>
              </div>
            </div>

            <div className="meta-grid">
              <div className="meta-item">
                <div className="meta-label">Width</div>
                <div className="meta-value">{imageMetadata.width?.toLocaleString()} px</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Height</div>
                <div className="meta-value">{imageMetadata.height?.toLocaleString()} px</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Bands</div>
                <div className="meta-value">{imageMetadata.bands}</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Chunks</div>
                <div className="meta-value">{imageMetadata.total_inference_chunks}</div>
              </div>
              <div className="meta-item span-2">
                <div className="meta-label">CRS</div>
                <div className="meta-value">{imageMetadata.crs}</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── Error ── */}
        {uploadStatus === 'error' && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            <div className="upload-error">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              Upload failed.
              <button className="btn-retry" onClick={reset}>Retry</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
