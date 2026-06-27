import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';

const FEATURE_COLORS = {
  building:   '#F97316',
  road:       '#2563EB',
  road_added: '#F59E0B',
  water:      '#06B6D4',
};

const FEATURE_DISPLAY = {
  building:   'Buildings',
  road:       'Roads',
  road_added: 'New Roads',
  water:      'Water Bodies',
};

export default function DetectionLegend() {
  const { detections, activeFilters } = useStore();

  const legendItems = useMemo(() => {
    const seen = new Map();
    detections.forEach((d) => {
      const key = d.display_label || d.feature_type;
      const base = d.base_feature_type || d.feature_type;
      if (!seen.has(key)) {
        seen.set(key, {
          color: d.colour || FEATURE_COLORS[base] || '#888',
          count: 0,
          label: FEATURE_DISPLAY[key] || FEATURE_DISPLAY[base] || key,
        });
      }
      seen.get(key).count++;
    });
    return Array.from(seen.entries())
      .filter(([key]) => activeFilters[key] !== false)
      .sort(([a], [b]) => a.localeCompare(b));
  }, [detections, activeFilters]);

  if (legendItems.length === 0) return null;

  return (
    <motion.div
      className="detection-legend"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="legend-title">Layer Legend</div>
      <div className="legend-items">
        {legendItems.map(([key, { color, count, label }]) => (
          <div key={key} className="legend-item">
            <div className="legend-swatch" style={{ background: color }} />
            <span className="legend-name">{label}</span>
            <span className="legend-count">{count.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
