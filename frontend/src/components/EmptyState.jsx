import React from 'react';
import { motion } from 'framer-motion';

export default function EmptyState() {
  return (
    <div className="empty-state">
      <motion.div
        className="empty-state-inner"
        initial={{ opacity: 0, scale: 0.96, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="empty-state-illustration">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" />
            <circle cx="12" cy="9" r="2.5" />
          </svg>
        </div>

        <div className="empty-state-title">
          Upload a GeoTIFF to Begin Analysis
        </div>
        <div className="empty-state-subtitle">
          AI-powered semantic segmentation detects buildings, roads, and
          water bodies from high-resolution drone orthophotos.
        </div>

        <div className="empty-state-tags">
          <div className="empty-tag building">
            <span className="empty-tag-dot" />
            Buildings
          </div>
          <div className="empty-tag road">
            <span className="empty-tag-dot" />
            Roads
          </div>
          <div className="empty-tag water">
            <span className="empty-tag-dot" />
            Water Bodies
          </div>
        </div>
      </motion.div>
    </div>
  );
}
