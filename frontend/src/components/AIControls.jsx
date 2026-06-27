import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { startInference } from '../services/api';
import { useStore } from '../store/useStore';

export default function AIControls() {
  const [loading, setLoading] = useState(false);

  const {
    uploadId, confidenceThreshold,
    jobStatus,
    setConfidenceThreshold, setJobId, setJobStatus,
  } = useStore();

  const isRunning = jobStatus === 'running' || jobStatus === 'started' || jobStatus === 'queued';

  const handleStart = async () => {
    if (!uploadId || isRunning || loading) return;
    setLoading(true);
    try {
      const { job_id } = await startInference(uploadId, confidenceThreshold);
      setJobId(job_id);
      setJobStatus('queued');
    } catch (err) {
      console.error('Failed to start inference:', err);
      alert(`Failed to start detection: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.05 }}
    >
      <div className="card-header">
        <div className="card-icon card-icon-green">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
        </div>
        <div>
          <div className="card-title">AI Settings</div>
          <div className="card-subtitle">SegFormer ONNX · RGB</div>
        </div>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span className="confidence-label">Min Confidence Threshold</span>
          <span className="confidence-value">{Math.round(confidenceThreshold * 100)}%</span>
        </div>
        <input
          type="range"
          className="slider"
          min="0.05"
          max="0.5"
          step="0.05"
          value={confidenceThreshold}
          onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
          disabled={isRunning || loading}
        />
        <div className="confidence-marks">
          <span className="confidence-mark">5%</span>
          <span className="confidence-mark">25%</span>
          <span className="confidence-mark">50%</span>
        </div>
      </div>

      <button
        className="btn-start-detection"
        onClick={handleStart}
        disabled={isRunning || loading || !uploadId}
      >
        {loading ? (
          <>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              style={{ animation: 'spin 1s linear infinite' }}>
              <path d="M12 2a10 10 0 0 1 10 10" />
            </svg>
            Starting…
          </>
        ) : (
          <>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Start AI Detection
          </>
        )}
      </button>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </motion.div>
  );
}
