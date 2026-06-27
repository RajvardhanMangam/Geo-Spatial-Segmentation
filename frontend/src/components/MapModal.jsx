import React, { useEffect, useMemo } from 'react';
import DetectionMap from './DetectionMap';
import { useStore } from '../store/useStore';
import { getGeoJsonUrl } from '../services/api';

const FEATURE_COLORS = {
  building:   '#FF4444',
  road:       '#4488FF',
  road_added: '#FFD23F',
  water:      '#00BBFF',
};

const FEATURE_LABELS = {
  building:   'Buildings',
  road:       'Roads',
  road_added: 'New Roads',
  water:      'Water',
};

export default function MapModal({ onClose }) {
  const { detections, activeFilters, toggleFilter, jobId, jobStatus, imageMetadata } = useStore();

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Build filter chips from actual detections (uses display_label as key, matching store)
  const filterChips = useMemo(() => {
    const seen = new Map();
    detections.forEach((d) => {
      const key = d.display_label || d.feature_type;
      if (!seen.has(key)) {
        seen.set(key, d.colour || FEATURE_COLORS[d.base_feature_type] || FEATURE_COLORS[d.feature_type] || '#888');
      }
    });
    return Array.from(seen.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [detections]);

  const isDone = jobStatus === 'completed';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        {/* Modal header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="modal-icon">◈</span>
            <span className="modal-title">Detection Map</span>
            {imageMetadata && (
              <span className="modal-meta">{imageMetadata.crs}</span>
            )}
          </div>

          <div className="modal-filters">
            {filterChips.map(([key, color]) => (
              <button
                key={key}
                className={`filter-chip ${activeFilters[key] !== false ? 'chip-on' : 'chip-off'}`}
                style={{ '--chip-color': color }}
                onClick={() => toggleFilter(key)}
                title={`Toggle ${key}`}
              >
                <span className="chip-swatch" />
                {FEATURE_LABELS[key] || key}
              </button>
            ))}
          </div>

          <div className="modal-actions">
            {isDone && jobId && (
              <a
                href={getGeoJsonUrl(jobId)}
                download
                className="btn-export-modal"
                title="Download GeoJSON"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Export
              </a>
            )}
            <button className="modal-close-btn" onClick={onClose} title="Close (Esc)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Map fills the rest */}
        <div className="modal-map-container">
          <DetectionMap />
        </div>
      </div>
    </div>
  );
}
