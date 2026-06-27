import React, { useMemo, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';

function useCountUp(target, duration = 600) {
  const [count, setCount] = useState(0);
  const rafRef = useRef(null);
  const startRef = useRef(null);
  const fromRef = useRef(0);

  useEffect(() => {
    fromRef.current = count;
    startRef.current = performance.now();
    const from = fromRef.current;

    const animate = (now) => {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.round(from + (target - from) * eased));
      if (progress < 1) rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target]);

  return count;
}

function StatCard({ icon, label, value, colorClass, delay = 0 }) {
  const count = useCountUp(value);
  return (
    <motion.div
      className="stat-card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
    >
      <div className={`stat-card-icon ${colorClass}`}>{icon}</div>
      <div className="stat-card-num">{count.toLocaleString()}</div>
      <div className="stat-card-label">{label}</div>
    </motion.div>
  );
}

export default function StatsCards() {
  const { detections } = useStore();

  const stats = useMemo(() => {
    let buildings = 0, roads = 0, water = 0;
    detections.forEach((d) => {
      const t = d.base_feature_type || d.feature_type || '';
      if (t === 'building')              buildings++;
      else if (t === 'road' || t === 'road_added') roads++;
      else if (t === 'water')            water++;
    });
    return { buildings, roads, water, total: detections.length };
  }, [detections]);

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="card-header">
        <div className="card-icon card-icon-cyan">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="3" width="6" height="8" rx="1" />
            <rect x="9" y="7" width="6" height="4" rx="1" />
            <rect x="16" y="5" width="6" height="6" rx="1" />
            <line x1="2" y1="19" x2="22" y2="19" />
          </svg>
        </div>
        <div>
          <div className="card-title">Detection Statistics</div>
          <div className="card-subtitle">{detections.length.toLocaleString()} total features</div>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard
          delay={0}
          colorClass="stat-icon-building"
          value={stats.buildings}
          label="Buildings"
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 22V9l9-7 9 7v13H3z" />
              <path d="M9 22v-7h6v7" />
            </svg>
          }
        />
        <StatCard
          delay={0.05}
          colorClass="stat-icon-road"
          value={stats.roads}
          label="Roads"
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 21l4-18h8l4 18" />
              <path d="M8 12h8" />
            </svg>
          }
        />
        <StatCard
          delay={0.1}
          colorClass="stat-icon-water"
          value={stats.water}
          label="Water Bodies"
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2C6 8 3 13 3 16a9 9 0 0 0 18 0c0-3-3-8-9-14z" />
            </svg>
          }
        />
        <StatCard
          delay={0.15}
          colorClass="stat-icon-total"
          value={stats.total}
          label="Total Objects"
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
          }
        />
      </div>
    </motion.div>
  );
}
