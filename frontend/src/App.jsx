import React from 'react';
import { AnimatePresence } from 'framer-motion';
import { useStore } from './store/useStore';
import { useJobStream } from './hooks/useJobStream';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import CesiumGlobe from './components/CesiumGlobe';
import DetectionLegend from './components/DetectionLegend';
import EmptyState from './components/EmptyState';
import StatusBar from './components/StatusBar';
import './App.css';

export default function App() {
  const { jobId, detections, imageMetadata, uploadStatus } = useStore();
  useJobStream(jobId);

  const isIdle = !imageMetadata && uploadStatus === 'idle';

  return (
    <div className="app">
      <Header />

      <div className="app-body">
        <Sidebar />

        <main className="map-area">
          {/* CesiumJS 3D globe — always rendered */}
          <CesiumGlobe />

          {/* Empty-state overlay (semi-transparent, globe visible behind) */}
          <AnimatePresence>
            {isIdle && <EmptyState key="empty" />}
          </AnimatePresence>

          {/* Floating detection legend */}
          <AnimatePresence>
            {detections.length > 0 && <DetectionLegend key="legend" />}
          </AnimatePresence>
        </main>
      </div>

      <StatusBar />
    </div>
  );
}
