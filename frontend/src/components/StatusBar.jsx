import React from 'react';
import { useStore } from '../store/useStore';

export default function StatusBar() {
  const { imageMetadata, detections, jobStatus, inferenceProgress } = useStore();

  const isRunning = jobStatus === 'running' || jobStatus === 'started';
  const isDone    = jobStatus === 'completed';

  return (
    <div className="statusbar">
      <div className="statusbar-segment">
        <span className="statusbar-dot green" />
        <span className="statusbar-label">Backend</span>
        <span className="statusbar-value">Connected</span>
      </div>

      <div className="statusbar-segment">
        <span className="statusbar-dot green" />
        <span className="statusbar-label">Model</span>
        <span className="statusbar-value">SegFormer ONNX</span>
      </div>

      <div className="statusbar-segment">
        <span className="statusbar-dot green" />
        <span className="statusbar-label">Redis</span>
        <span className="statusbar-value">Connected</span>
      </div>

      {isRunning && (
        <div className="statusbar-segment">
          <span className="statusbar-dot blue" />
          <span className="statusbar-label">Inference</span>
          <span className="statusbar-value mono">{inferenceProgress.toFixed(1)}%</span>
        </div>
      )}

      {imageMetadata && (
        <>
          <div className="statusbar-segment">
            <span className="statusbar-label">CRS</span>
            <span className="statusbar-value mono">{imageMetadata.crs}</span>
          </div>
          <div className="statusbar-segment">
            <span className="statusbar-label">Size</span>
            <span className="statusbar-value mono">
              {imageMetadata.width?.toLocaleString()} × {imageMetadata.height?.toLocaleString()}
            </span>
          </div>
          <div className="statusbar-segment">
            <span className="statusbar-label">Chunks</span>
            <span className="statusbar-value mono">{imageMetadata.total_inference_chunks}</span>
          </div>
        </>
      )}

      {detections.length > 0 && (
        <div className="statusbar-segment">
          <span className="statusbar-label">Detections</span>
          <span className="statusbar-value mono">{detections.length.toLocaleString()}</span>
        </div>
      )}

      <div className="statusbar-spacer" />

      <div className="statusbar-segment" style={{ borderRight: 'none' }}>
        <span className="statusbar-label">
          GeoSight AI · MoPR Rural Intelligence Platform · SegFormer v2
        </span>
      </div>
    </div>
  );
}
