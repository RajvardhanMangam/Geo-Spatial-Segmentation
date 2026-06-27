import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import { globeAPI } from '../utils/globeRef';
import { getGeoJsonUrl } from '../services/api';

/* ── Settings Drawer ─────────────────────────────────────── */
function SettingsDrawer({ rotationOn, onToggleRotation, onClose }) {
  return (
    <div className="settings-drawer" onClick={(e) => e.stopPropagation()}>
      <div className="settings-drawer-header">
        <span className="settings-drawer-title">Globe Settings</span>
        <button className="settings-close-btn" onClick={onClose} aria-label="Close settings">
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8" width="12" height="12">
            <path d="M2 2l10 10M12 2L2 12" />
          </svg>
        </button>
      </div>

      <div className="settings-group">
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-label">Globe Rotation</span>
            <span className="settings-desc">Auto-rotate in idle state</span>
          </div>
          <button
            className={`settings-toggle ${rotationOn ? 'settings-toggle--on' : ''}`}
            onClick={onToggleRotation}
            aria-label="Toggle globe rotation"
          >
            <span className="settings-toggle-knob" />
          </button>
        </div>

        <div className="settings-row settings-row--static">
          <div className="settings-row-info">
            <span className="settings-label">Imagery</span>
            <span className="settings-desc">ESRI World Imagery</span>
          </div>
          <span className="settings-tag">Active</span>
        </div>

        <div className="settings-row settings-row--static">
          <div className="settings-row-info">
            <span className="settings-label">Renderer</span>
            <span className="settings-desc">CesiumJS WebGL</span>
          </div>
          <span className="settings-tag">GPU</span>
        </div>

        <div className="settings-row settings-row--static">
          <div className="settings-row-info">
            <span className="settings-label">Terrain</span>
            <span className="settings-desc">Ellipsoid (flat)</span>
          </div>
          <span className="settings-tag">Default</span>
        </div>

        <div className="settings-row settings-row--static">
          <div className="settings-row-info">
            <span className="settings-label">Model</span>
            <span className="settings-desc">SegFormer ONNX · RGB · 1024 px</span>
          </div>
          <span className="settings-tag">Loaded</span>
        </div>
      </div>

      <div className="settings-footer">
        GeoSight AI · MoPR v2.0
      </div>
    </div>
  );
}

/* ── Header ──────────────────────────────────────────────── */
export default function Header() {
  const { jobStatus, inferenceProgress, imageMetadata, jobId } = useStore();
  const [settingsOpen, setSettingsOpen]   = useState(false);
  const [rotationOn,   setRotationOn]     = useState(true);

  const hasArea   = !!imageMetadata;
  const canExport = jobStatus === 'completed' && !!jobId;
  const isRunning = ['queued', 'started', 'running'].includes(jobStatus);

  const handleGlobalView = () => {
    globeAPI.flyToGlobe();
    setRotationOn(true);
  };

  const handleGoToArea = () => globeAPI.flyToArea();
  const handleFitBounds = () => globeAPI.fitBounds();
  const handleResetCamera = () => globeAPI.resetCamera();

  const handleExportGeoJSON = () => {
    if (!jobId) return;
    const a = document.createElement('a');
    a.href = getGeoJsonUrl(jobId);
    a.download = `detections_${jobId.slice(0, 8)}.geojson`;
    a.click();
  };

  const handleToggleRotation = () => {
    const next = !rotationOn;
    setRotationOn(next);
    if (globeAPI.rotRef) globeAPI.rotRef.current = next;
  };

  return (
    <>
      <header className="header">
        {/* ── Brand ── */}
        <div className="header-brand">
          <div className="header-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 3C12 3 9 7 9 12C9 17 12 21 12 21" />
              <path d="M12 3C12 3 15 7 15 12C15 17 12 21 12 21" />
              <path d="M3.05 9h17.9M3.05 15h17.9" />
            </svg>
          </div>
          <div className="header-brand-text">
            <span className="header-brand-name">
              GeoSight
              <span className="header-ai-badge">AI</span>
            </span>
            <span className="header-brand-sub">MoPR Rural Mapping Dashboard</span>
          </div>
        </div>

        <div className="header-divider" />

        {/* ── Navigation toolbar ── */}
        <nav className="header-nav">
          <button className="header-nav-btn" onClick={handleGlobalView} title="Fly back to world view">
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
              <circle cx="10" cy="10" r="8" />
              <path d="M10 2c0 0-2.5 3.5-2.5 8s2.5 8 2.5 8" />
              <path d="M10 2c0 0 2.5 3.5 2.5 8s-2.5 8-2.5 8" />
              <path d="M2.1 7.5h15.8M2.1 12.5h15.8" />
            </svg>
            Global View
          </button>

          {hasArea && (
            <>
              <button className="header-nav-btn" onClick={handleGoToArea} title="Fly to uploaded orthophoto">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
                  <path d="M10 2C6.69 2 4 4.69 4 8c0 4.5 6 10 6 10s6-5.5 6-10c0-3.31-2.69-6-6-6z" />
                  <circle cx="10" cy="8" r="2" />
                </svg>
                Go To Area
              </button>

              <button className="header-nav-btn" onClick={handleFitBounds} title="Fit camera to analysis bounds">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
                  <rect x="3" y="3" width="14" height="14" rx="1.5" />
                  <path d="M3 7h2M3 13h2M17 7h-2M17 13h-2M7 3v2M13 3v2M7 17v-2M13 17v-2" />
                </svg>
                Fit Bounds
              </button>
            </>
          )}

          <button className="header-nav-btn" onClick={handleResetCamera} title="Reset camera pitch and heading">
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
              <path d="M3.5 3.5L7 7M3.5 3.5H7M3.5 3.5V7" />
              <path d="M16.5 16.5L13 13M16.5 16.5H13M16.5 16.5V13" />
              <circle cx="10" cy="10" r="3" />
              <path d="M10 3v1M10 16v1M3 10h1M16 10h1" />
            </svg>
            Reset Camera
          </button>

          {canExport && (
            <button className="header-nav-btn header-nav-btn--export" onClick={handleExportGeoJSON} title="Download detections as GeoJSON">
              <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
                <path d="M10 3v10M6 9l4 4 4-4" />
                <path d="M3 15v1a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-1" />
              </svg>
              Export GeoJSON
            </button>
          )}

          <div className="header-nav-sep" />

          <button
            className={`header-nav-btn ${settingsOpen ? 'header-nav-btn--active' : ''}`}
            onClick={() => setSettingsOpen((s) => !s)}
            title="Globe settings"
          >
            <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" width="13" height="13">
              <circle cx="10" cy="10" r="2.5" />
              <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M3.7 3.7l1.4 1.4M14.9 14.9l1.4 1.4M3.7 16.3l1.4-1.4M14.9 5.1l1.4-1.4" />
            </svg>
            Settings
          </button>
        </nav>

        {/* ── Right: live status + MoPR badge ── */}
        <div className="header-right">
          {isRunning && (
            <div className="status-badge processing">
              <span className="status-badge-dot" />
              AI Running · {inferenceProgress.toFixed(0)}%
            </div>
          )}
          <div className="header-mopr-badge">
            <span className="header-mopr-dot" />
            MoPR HACKATHON
          </div>
        </div>
      </header>

      {/* Settings overlay + drawer (overlay starts below header so nav buttons remain clickable) */}
      {settingsOpen && (
        <>
          <div
            style={{
              position: 'fixed',
              top: 'var(--header-h)',
              left: 0, right: 0, bottom: 0,
              zIndex: 250,
            }}
            onClick={() => setSettingsOpen(false)}
          />
          <SettingsDrawer
            rotationOn={rotationOn}
            onToggleRotation={handleToggleRotation}
            onClose={() => setSettingsOpen(false)}
          />
        </>
      )}
    </>
  );
}
