import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/useStore';
import { getGeoJsonUrl } from '../services/api';

const RADIUS = 36;
const CIRC   = 2 * Math.PI * RADIUS;

function formatTime(s) {
  if (!s || s <= 0) return '—';
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

function StatusPill({ status }) {
  const label = {
    queued:    'Queued',
    running:   'Running',
    started:   'Running',
    completed: 'Completed',
    failed:    'Failed',
  }[status] || status;

  const cls = status === 'completed' ? 'completed'
            : status === 'failed'    ? 'failed'
            : status === 'queued'    ? 'queued'
            : 'running';

  return (
    <div className={`job-status-pill ${cls}`}>
      <span className="job-status-dot" />
      {label}
    </div>
  );
}

export default function JobProgress() {
  const {
    jobStatus, inferenceProgress, imageMetadata,
    detections, jobId, reset,
  } = useStore();

  const [startTime, setStartTime]   = useState(null);
  const [elapsed, setElapsed]       = useState(0);

  const isRunning  = jobStatus === 'running' || jobStatus === 'started' || jobStatus === 'queued';
  const isDone     = jobStatus === 'completed';
  const pct        = Math.min(100, Math.max(0, inferenceProgress));

  useEffect(() => {
    if (isRunning && !startTime) setStartTime(Date.now());
    if (isDone)                  setStartTime((prev) => prev); // keep
  }, [isRunning, isDone]);

  useEffect(() => {
    if (!isRunning) return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - (startTime || Date.now())) / 1000)), 1000);
    return () => clearInterval(t);
  }, [isRunning, startTime]);

  const estimatedRemaining = useMemo(() => {
    if (!isRunning || pct <= 0 || elapsed <= 0) return null;
    return Math.round((elapsed / pct) * (100 - pct));
  }, [isRunning, pct, elapsed]);

  const totalChunks = imageMetadata?.total_inference_chunks ?? 0;
  const chunksDone  = totalChunks > 0 ? Math.round((pct / 100) * totalChunks) : 0;

  const dashOffset = CIRC * (1 - pct / 100);
  const circleCls  = isDone ? 'completed' : isRunning ? 'running' : 'queued';

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="card-header">
        <div className="card-icon card-icon-green">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M6.3 6.3a8 8 0 1 0 11.31 0" />
          </svg>
        </div>
        <div>
          <div className="card-title">AI Processing</div>
          <div className="card-subtitle">Live inference stream</div>
        </div>
      </div>

      <StatusPill status={jobStatus} />

      {/* Circle + info */}
      <div className="progress-circle-wrap">
        <svg
          className="progress-circle-svg"
          width={RADIUS * 2 + 16}
          height={RADIUS * 2 + 16}
          viewBox={`0 0 ${RADIUS * 2 + 16} ${RADIUS * 2 + 16}`}
        >
          <circle
            className="progress-circle-bg"
            cx={RADIUS + 8}
            cy={RADIUS + 8}
            r={RADIUS}
          />
          <circle
            className={`progress-circle-fill ${circleCls}`}
            cx={RADIUS + 8}
            cy={RADIUS + 8}
            r={RADIUS}
            strokeDasharray={CIRC}
            strokeDashoffset={dashOffset}
          />
        </svg>

        <div className="progress-circle-info">
          <div className="progress-circle-pct">{pct.toFixed(0)}%</div>
          <div className="progress-circle-label">
            {totalChunks > 0 ? `${chunksDone} / ${totalChunks} chunks` : 'coverage'}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="inference-progress">
        <div className="progress-bar-header">
          <span className="progress-bar-label">Coverage</span>
          <span className="progress-bar-pct" style={{ color: isDone ? 'var(--primary)' : 'var(--secondary)' }}>
            {pct.toFixed(1)}%
          </span>
        </div>
        <div className="progress-track">
          <div
            className={`progress-fill ${isDone ? 'fill-done' : 'fill-inference'}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Time grid */}
      <div className="job-time-row">
        <div className="job-time-item">
          <div className="job-time-label">Elapsed</div>
          <div className="job-time-value">{formatTime(elapsed)}</div>
        </div>
        <div className="job-time-item">
          <div className="job-time-label">Remaining</div>
          <div className="job-time-value">{isDone ? '0s' : formatTime(estimatedRemaining)}</div>
        </div>
      </div>

      {/* Detection count */}
      <AnimatePresence>
        {detections.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              textAlign: 'center',
              padding: '10px 0 4px',
              borderTop: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <div style={{
              fontFamily: 'var(--font-head)',
              fontSize: 32,
              fontWeight: 800,
              color: 'var(--primary)',
              lineHeight: 1,
              letterSpacing: '-1px',
            }}>
              {detections.length.toLocaleString()}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, letterSpacing: 1, textTransform: 'uppercase' }}>
              features detected
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {isDone && jobId && (
          <a
            href={getGeoJsonUrl(jobId)}
            download
            className="btn-export-geojson"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export GeoJSON
          </a>
        )}
        <button
          onClick={reset}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            width: '100%', padding: '9px 12px',
            background: 'rgba(239,68,68,0.06)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: 'var(--radius-sm)',
            color: 'rgba(239,68,68,0.7)',
            fontSize: 11.5, fontWeight: 600, cursor: 'pointer',
            transition: 'all 0.15s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.14)'; e.currentTarget.style.color = '#EF4444'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239,68,68,0.06)'; e.currentTarget.style.color = 'rgba(239,68,68,0.7)'; }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 .49-3.51" />
          </svg>
          {isDone ? 'Start New Analysis' : 'Cancel & Reset'}
        </button>
      </div>
    </motion.div>
  );
}
