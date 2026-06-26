import React, { useRef } from 'react';
import { useStore } from './store/useStore';
import { useJobStream } from './hooks/useJobStream';
import CesiumGlobe from './components/CesiumGlobe';
import UploadPanel from './components/UploadPanel';
import ProgressPanel from './components/ProgressPanel';
import './App.css';

export default function App() {
  const {
    jobId, jobStatus, detections, inferenceProgress,
    activeFilters, toggleFilter,
  } = useStore();

  const cesiumRef = useRef(null);

  // Show completed if backend returned failed but detections exist
  const chipStatus = (jobStatus === 'failed' && detections.length > 0) ? 'completed' : jobStatus;

  useJobStream(jobId);

  return (
    <div className="app">
      {/* ── Topbar ────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="2" y1="12" x2="22" y2="12"/>
              <path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/>
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">Rural<em>Intel</em></span>
            <span className="brand-sub">Geospatial Analysis</span>
          </div>
        </div>

        {/* Live status chip */}
        <div className="topbar-center">
          {jobId && (
            <div className={`live-chip ${chipStatus === 'running' ? 'live-chip--active' : chipStatus === 'completed' ? 'live-chip--done' : ''}`}>
              <span className="live-dot" />
              {chipStatus === 'running'
                ? `Processing · ${inferenceProgress.toFixed(1)}%`
                : chipStatus === 'completed'
                ? `Complete · ${detections.length.toLocaleString()} features`
                : (chipStatus === 'queued' ? 'Queued' : chipStatus === 'failed' ? 'Failed' : chipStatus)}
            </div>
          )}
        </div>

        {/* Toolbar */}
        <nav className="topbar-nav">
          <button
            className="nav-action"
            onClick={() => cesiumRef.current?.resetCamera()}
            title="Return to global view"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="2" y1="12" x2="22" y2="12"/>
              <path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/>
            </svg>
            Global
          </button>
          <button
            className="nav-action"
            onClick={() => cesiumRef.current?.flyToVillage()}
            disabled={!jobId}
            title="Fly to uploaded site"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="3 11 22 2 13 21 11 13 3 11"/>
            </svg>
            Fly to Site
          </button>

          <div className="nav-divider" />

          {[
            { type: 'building', label: 'Buildings', color: '#CF7A3E' },
            { type: 'road',     label: 'Roads',     color: '#4E8AB0' },
            { type: 'water',    label: 'Water',     color: '#3EACB0' },
          ].map(({ type, label, color }) => (
            <button
              key={type}
              className={`nav-layer ${activeFilters[type] ? 'nav-layer--on' : ''}`}
              onClick={() => toggleFilter(type)}
              title={`Toggle ${label}`}
            >
              <span className="layer-dot" style={{ background: color }} />
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── Workspace ─────────────────────────────────────────────── */}
      <div className="workspace">
        <aside className="command-panel">
          <UploadPanel />
          <ProgressPanel />
          <div className="panel-footer">
            <span>SegFormer ONNX · 1024 px tiles</span>
            <a
              href="https://geo.intel.iittnif.com/activitiesinitiatives/mopr-hackathon"
              target="_blank"
              rel="noreferrer"
              className="footer-link"
            >
              MoPR ↗
            </a>
          </div>
        </aside>

        <main className="globe-wrap">
          <CesiumGlobe cesiumRef={cesiumRef} />
        </main>
      </div>
    </div>
  );
}
