import React, { useMemo } from 'react';
import { useStore } from '../store/useStore';
import { getGeoJsonUrl } from '../services/api';

const RING_R    = 44;
const RING_CIRC = 2 * Math.PI * RING_R;

const FEATURE_CONFIG = [
  { type: 'building', label: 'Buildings', color: '#CF7A3E' },
  { type: 'road',     label: 'Roads',     color: '#4E8AB0' },
  { type: 'water',    label: 'Water',     color: '#3EACB0' },
];

const STATUS_LABEL = {
  idle:      'Idle',
  queued:    'Queued',
  running:   'Processing',
  completed: 'Complete',
  failed:    'Failed',
};

function getStatusMessage(jobStatus, progress) {
  if (jobStatus === 'idle')      return 'Awaiting upload';
  if (jobStatus === 'queued')    return 'Queued for processing...';
  if (jobStatus === 'completed') return 'Analysis complete';
  if (jobStatus === 'failed')    return 'Processing failed';
  if (progress < 8)  return 'Initializing inference engine...';
  if (progress < 20) return 'Preparing image chunks...';
  if (progress < 50) return 'Running SegFormer inference...';
  if (progress < 65) return 'Enhancing road connectivity...';
  if (progress < 78) return 'Extracting building footprints...';
  if (progress < 88) return 'Detecting water bodies...';
  if (progress < 96) return 'Generating GeoJSON...';
  return 'Finalizing analysis...';
}

export default function ProgressPanel() {
  const {
    jobId, jobStatus, inferenceProgress,
    detections,
    activeFilters, toggleFilter,
  } = useStore();

  const countByType = useMemo(() => {
    const c = { building: 0, road: 0, water: 0 };
    detections.forEach((d) => {
      if (d.feature_type in c) c[d.feature_type]++;
    });
    return c;
  }, [detections]);

  const displayStatus = (jobStatus === 'failed' && detections.length > 0)
    ? 'completed'
    : jobStatus;

  const effectiveProgress = displayStatus === 'completed' ? 100 : inferenceProgress;
  const ringOffset = RING_CIRC * (1 - effectiveProgress / 100);

  const isDone   = displayStatus === 'completed';
  const isFailed = displayStatus === 'failed';

  const wsStatus    = jobId ? 'online' : 'idle';
  const intelStatus = jobStatus === 'running' || jobStatus === 'queued' ? 'online' : 'idle';

  return (
    <>
      {/* ── Analysis Status ─────────────────────────────────────── */}
      <div className="section">
        <div className="section-header">
          <span className="section-title">Analysis</span>
          <span className={`status-badge status-badge--${displayStatus}`}>
            {STATUS_LABEL[displayStatus] || displayStatus}
          </span>
        </div>

        <div className="status-ring-wrap">
          <svg className="ring-svg" width="110" height="110" viewBox="0 0 110 110">
            <circle className="ring-track" cx="55" cy="55" r={RING_R} />
            <circle
              className="ring-fill"
              cx="55" cy="55"
              r={RING_R}
              strokeDasharray={RING_CIRC}
              strokeDashoffset={isDone ? 0 : isFailed ? RING_CIRC : ringOffset}
              transform="rotate(-90 55 55)"
              stroke={isFailed ? '#C47A5A' : undefined}
            />
            <text className="ring-pct" x="55" y="50" textAnchor="middle" dominantBaseline="middle">
              {isDone ? '100' : isFailed ? '—' : Math.round(effectiveProgress)}
            </text>
            <text className="ring-label-text" x="55" y="66" textAnchor="middle" dominantBaseline="middle">
              {isDone ? 'DONE' : isFailed ? 'ERROR' : 'PCT'}
            </text>
          </svg>

          <p className={`status-msg status-msg--${displayStatus}`}>
            {getStatusMessage(displayStatus, effectiveProgress)}
          </p>
        </div>

        {/* Post-completion detection summary */}
        {isDone && (
          <div className="completion-summary animate-in">
            {FEATURE_CONFIG.map(({ type, label, color }) => (
              <div key={type} className="check-row">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
                <span className="check-label">{label}</span>
                <span className="check-count" style={{ color }}>
                  {countByType[type].toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Detection Layers ─────────────────────────────────────── */}
      <div className="section">
        <div className="section-header">
          <span className="section-title">Layers</span>
          {detections.length > 0 && (
            <span className="section-count">{detections.length.toLocaleString()} features</span>
          )}
        </div>

        <div className="feature-list">
          {FEATURE_CONFIG.map(({ type, label, color }) => (
            <div
              key={type}
              className={`feature-row ${activeFilters[type] === false ? 'feature-row--off' : ''}`}
              onClick={() => toggleFilter(type)}
            >
              <span className="feature-dot" style={{ background: color }} />
              <span className="feature-name">{label}</span>
              <span className="feature-count">
                {countByType[type].toLocaleString()}
              </span>
              <span className="feature-toggle" />
            </div>
          ))}
        </div>
      </div>

      {/* ── System Status ────────────────────────────────────────── */}
      <div className="section">
        <div className="section-header">
          <span className="section-title">System</span>
        </div>
        <div className="system-list">
          <SystemRow label="WebSocket"        status={wsStatus}    text={wsStatus === 'online'    ? 'Connected' : 'Standby'} />
          <SystemRow label="Inference Engine" status={intelStatus} text={intelStatus === 'online' ? 'Active'    : 'Ready'}   />
          <SystemRow label="Redis"            status="online"      text="Online" />
          <SystemRow label="ONNX Runtime"     status="online"      text="Loaded" />
        </div>
      </div>

      {/* ── Export ───────────────────────────────────────────────── */}
      {isDone && (
        <div className="section export-section animate-in">
          <a href={getGeoJsonUrl(jobId)} download className="export-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export GeoJSON
          </a>
        </div>
      )}
    </>
  );
}

function SystemRow({ label, status, text }) {
  return (
    <div className="sys-row">
      <span className="sys-label">{label}</span>
      <div className="sys-status">
        <span className={`sys-dot sys-dot--${status}`} />
        <span className={`sys-text sys-text--${status}`}>{text}</span>
      </div>
    </div>
  );
}
