import React, { useMemo } from 'react';
import { useStore } from '../store/useStore';
import { getGeoJsonUrl } from '../services/api';

const FEATURE_COLORS = {
  building:   '#FF4444',
  road:       '#4488FF',
  road_added: '#FFD23F',
  water:      '#00BBFF',
};

export default function ProgressPanel({ onViewMap }) {
  const {
    jobId, jobStatus, inferenceProgress,
    detections, imageMetadata,
    activeFilters, toggleFilter,
  } = useStore();

  const countByLabel = useMemo(() => {
    const counts = {};
    detections.forEach((d) => {
      const key = d.display_label || d.feature_type;
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }, [detections]);

  const featureRows = useMemo(() => {
    const colorMap = {};
    detections.forEach((d) => {
      const key = d.display_label || d.feature_type;
      if (!colorMap[key]) {
        colorMap[key] = d.colour || FEATURE_COLORS[d.base_feature_type] || FEATURE_COLORS[d.feature_type] || '#888';
      }
    });
    return Object.entries(colorMap).sort(([a], [b]) => a.localeCompare(b));
  }, [detections]);

  if (!jobId) return null;

  const isRunning = jobStatus === 'running' || jobStatus === 'queued' || jobStatus === 'started';
  const isDone    = jobStatus === 'completed';
  const pct       = inferenceProgress;

  return (
    <div className="panel">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M6.3 6.3a8 8 0 1 0 11.31 0"/>
            <path d="M12 2v2"/>
          </svg>
        </div>
        <div className="panel-label">
          <div className="panel-title">Analysis</div>
          <div className="panel-subtitle">
            {isDone ? 'Completed' : isRunning ? 'Processing…' : jobStatus}
          </div>
        </div>
        {isDone && (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2.5">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        )}
      </div>

      {/* Big detection count */}
      <div className="detection-count-block">
        <div className="detection-count-num">{detections.length.toLocaleString()}</div>
        <div className="detection-count-label">features detected</div>
      </div>

      {/* Progress bar */}
      <div style={{ marginBottom: 16 }}>
        <div className="progress-label-row">
          <span className="progress-label-text">Coverage</span>
          <span className="progress-label-pct" style={{ color: isDone ? 'var(--green)' : 'var(--cyan)' }}>
            {pct.toFixed(1)}%
          </span>
        </div>
        <div className="scan-track">
          <div
            className={`progress-fill ${isDone ? 'fill-done' : 'fill-progress'}`}
            style={{ width: `${pct}%` }}
          />
          {isRunning && (
            <div className="scan-glow" style={{ left: `${pct}%` }} />
          )}
        </div>
      </div>

      {/* Feature breakdown */}
      {featureRows.length > 0 && (
        <div className="feature-list">
          {featureRows.map(([key, color]) => {
            const active = activeFilters[key] !== false;
            return (
              <div
                key={key}
                className={`feature-row ${active ? '' : 'row-inactive'}`}
                onClick={() => toggleFilter(key)}
                title={`${active ? 'Hide' : 'Show'} ${key}`}
              >
                <div className="feature-swatch" style={{ background: color }} />
                <span className="feature-name">{key}</span>
                <span className="feature-count">{(countByLabel[key] || 0).toLocaleString()}</span>
                <div className="feature-toggle">{active ? '●' : '○'}</div>
              </div>
            );
          })}
        </div>
      )}

      {/* Action buttons */}
      <div className="panel-actions">
        {isDone && (
          <button className="btn-view-map" onClick={onViewMap}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="3 11 22 2 13 21 11 13 3 11"/>
            </svg>
            View Detections on Map
          </button>
        )}

        {isDone && jobId && (
          <a href={getGeoJsonUrl(jobId)} download className="btn-export">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export GeoJSON
          </a>
        )}
      </div>

      {/* Footer metadata */}
      {imageMetadata && (
        <div className="chunks-footer">
          <span>{imageMetadata.crs}</span>
          <span>{imageMetadata.total_inference_chunks} chunks</span>
        </div>
      )}
    </div>
  );
}
