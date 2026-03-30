/**
 * MoPR Rural Feature Detector — Main App
 */
import React from 'react';
import { useStore } from './store/useStore';
import { useJobStream } from './hooks/useJobStream';
import DetectionMap from './components/DetectionMap';
import UploadPanel from './components/UploadPanel';
import ProgressPanel from './components/ProgressPanel';
import './App.css';

export default function App() {
  const { jobId, jobStatus, detections, inferenceProgress } = useStore();

  // Connect WebSocket when a job is active
  useJobStream(jobId);

  return (
    <div className="app">
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-left">
          <span className="logo-mark">◈</span>
          <span className="logo-text">MoPR <em>Rural Detector</em></span>
          <span className="logo-sub">Hackathon · Problem Statement 1</span>
        </div>
        <div className="topbar-right">
          {jobId && (
            <div className="live-indicator">
              <span className={`pulse ${jobStatus === 'running' ? 'pulse-active' : ''}`} />
              <span className="live-text">
                {jobStatus === 'running'
                  ? `Processing · ${inferenceProgress.toFixed(1)}%`
                  : jobStatus === 'completed'
                  ? `Done · ${detections.length} features`
                  : jobStatus}
              </span>
            </div>
          )}
          <a
            href="https://geo.intel.iittnif.com/activitiesinitiatives/mopr-hackathon"
            target="_blank"
            rel="noreferrer"
            className="hackathon-link"
          >
            MoPR Hackathon ↗
          </a>
        </div>
      </header>

      {/* Main layout */}
      <div className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <UploadPanel />
          <ProgressPanel />
          <div className="sidebar-footer">
            <span>Mask2Former · 4-band · 1024px chunks</span>
          </div>
        </aside>

        {/* Map canvas */}
        <main className="map-canvas">
          <DetectionMap />

          {/* No-job overlay */}
          {!jobId && (
            <div className="map-overlay">
              <div className="overlay-content">
                <div className="overlay-icon">◈</div>
                <div className="overlay-title">Upload a drone orthophoto</div>
                <div className="overlay-sub">
                  Supports GeoTIFF · EPSG:32644 / 3857 · up to 6 GB
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
