/**
 * ProgressPanel — shows inference progress, live detection counts,
 * and feature type breakdown.
 */
import React, { useMemo } from 'react';
import { useStore } from '../store/useStore';
import { getGeoJsonUrl } from '../services/api';

const FEATURE_COLORS = {
  building:   '#FF4444',
  road:       '#4488FF',
  road_added: '#FFD23F',
  water:      '#00BBFF',
};

export default function ProgressPanel() {
  const {
    jobId, jobStatus, inferenceProgress,
    detections, imageMetadata,
    activeFilters, toggleFilter,
  } = useStore();

  const countByType = useMemo(() => {
    const counts = {};
    detections.forEach((d) => {
      const label = d.display_label || d.feature_type;
      counts[label] = (counts[label] || 0) + 1;
    });
    return counts;
  }, [detections]);

  const featureRows = useMemo(() => {
    const colors = {};
    detections.forEach((d) => {
      const label = d.display_label || d.feature_type;
      if (!colors[label]) {
        colors[label] =
          d.colour || FEATURE_COLORS[d.base_feature_type] || FEATURE_COLORS[d.feature_type] || '#888888';
      }
    });

    return Object.entries(colors).sort(([a], [b]) => a.localeCompare(b));
  }, [detections]);

  if (!jobId) return null;

  const isRunning = jobStatus === 'running' || jobStatus === 'queued';
  const isDone = jobStatus === 'completed';

  return (
    <div className="progress-panel">
      <div className="panel-header">
        <span className="panel-title">INFERENCE</span>
        <span className={`status-badge status-${jobStatus}`}>{jobStatus}</span>
      </div>

      {/* Progress bar */}
      <div className="progress-section">
        <div className="progress-label-row">
          <span>Village processed</span>
          <span className="progress-pct">{inferenceProgress.toFixed(1)}%</span>
        </div>
        <div className="progress-bar-track">
          <div
            className={`progress-bar-fill ${isDone ? 'done-fill' : 'inference-fill'}`}
            style={{ width: `${inferenceProgress}%` }}
          />
        </div>
        {isRunning && (
          <div className="scanning-line" style={{ left: `${inferenceProgress}%` }} />
        )}
      </div>

      {/* Detection counts */}
      <div className="count-total">
        <span className="count-num">{detections.length.toLocaleString()}</span>
        <span className="count-label">features detected</span>
      </div>

      {/* Per-type breakdown + toggles */}
      <div className="feature-list">
        {featureRows.map(([type, color]) => (
          <div
            key={type}
            className={`feature-row ${activeFilters[type] !== false ? 'active' : 'inactive'}`}
            onClick={() => toggleFilter(type)}
          >
            <span className="feature-dot" style={{ background: color }} />
            <span className="feature-name">{type}</span>
            <span className="feature-count">{countByType[type] || 0}</span>
            <span className="feature-toggle">{activeFilters[type] !== false ? '●' : '○'}</span>
          </div>
        ))}
      </div>

      {/* Export */}
      {isDone && (
        <a
          href={getGeoJsonUrl(jobId)}
          download
          className="export-btn"
        >
          ↓ Export GeoJSON
        </a>
      )}

      {/* Image metadata summary */}
      {imageMetadata && (
        <div className="meta-footer">
          <span>{imageMetadata.crs}</span>
          <span>{imageMetadata.total_inference_chunks} chunks</span>
        </div>
      )}
    </div>
  );
}
