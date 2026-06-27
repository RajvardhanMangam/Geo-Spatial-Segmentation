import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';

const FEATURE_COLORS = {
  building:   '#F97316',
  road:       '#2563EB',
  road_added: '#F59E0B',
  water:      '#06B6D4',
};

export default function DetectionFilters() {
  const { detections, activeFilters, toggleFilter } = useStore();

  const featureRows = useMemo(() => {
    const seen = new Map();
    detections.forEach((d) => {
      const key = d.display_label || d.feature_type;
      if (!seen.has(key)) {
        seen.set(key, {
          color: d.colour || FEATURE_COLORS[d.base_feature_type] || FEATURE_COLORS[d.feature_type] || '#888',
          count: 0,
        });
      }
      seen.get(key).count++;
    });
    return Array.from(seen.entries())
      .sort(([a], [b]) => a.localeCompare(b));
  }, [detections]);

  if (featureRows.length === 0) return null;

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="card-header">
        <div className="card-icon card-icon-blue">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="4" y1="6" x2="20" y2="6" />
            <line x1="8" y1="12" x2="16" y2="12" />
            <line x1="10" y1="18" x2="14" y2="18" />
          </svg>
        </div>
        <div>
          <div className="card-title">Detection Filters</div>
          <div className="card-subtitle">Toggle layer visibility</div>
        </div>
      </div>

      <div className="filter-list">
        {featureRows.map(([key, { color, count }], i) => {
          const isActive = activeFilters[key] !== false;
          return (
            <motion.div
              key={key}
              className={`filter-row${isActive ? '' : ' inactive'}`}
              onClick={() => toggleFilter(key)}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2, delay: i * 0.04 }}
            >
              <div
                className="filter-swatch"
                style={{ background: color, opacity: isActive ? 1 : 0.4 }}
              />
              <span className="filter-name">{key}</span>
              <span className="filter-count">{count.toLocaleString()}</span>
              <label className="filter-toggle" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={() => toggleFilter(key)}
                  readOnly
                />
                <span className="filter-toggle-track" />
              </label>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
