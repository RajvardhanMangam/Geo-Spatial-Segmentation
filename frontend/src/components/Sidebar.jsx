import React from 'react';
import { useStore } from '../store/useStore';
import UploadCard from './UploadCard';
import AIControls from './AIControls';
import JobProgress from './JobProgress';
import StatsCards from './StatsCards';
import DetectionFilters from './DetectionFilters';

export default function Sidebar() {
  const { uploadStatus, imageMetadata, jobId, detections } = useStore();

  const hasImage       = uploadStatus === 'complete' && !!imageMetadata;
  const hasJob         = !!jobId;
  const hasDetections  = detections.length > 0;

  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <UploadCard />
      </div>

      {hasImage && !hasJob && (
        <div className="sidebar-section">
          <AIControls />
        </div>
      )}

      {hasJob && (
        <div className="sidebar-section">
          <JobProgress />
        </div>
      )}

      {hasDetections && (
        <div className="sidebar-section">
          <StatsCards />
        </div>
      )}

      {hasDetections && (
        <div className="sidebar-section">
          <DetectionFilters />
        </div>
      )}

      <div className="sidebar-footer">
        <span className="sidebar-footer-text">SegFormer ONNX · RGB · 1024 px</span>
        <span className="sidebar-footer-text">MoPR · v2.0</span>
      </div>
    </aside>
  );
}
