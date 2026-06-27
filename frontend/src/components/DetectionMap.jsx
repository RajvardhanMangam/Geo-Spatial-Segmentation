/**
 * DetectionMap — full-screen Leaflet map
 * Logic unchanged; visual styling updated to match enterprise dashboard.
 */
import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import {
  MapContainer,
  TileLayer,
  Polygon,
  Popup,
  Rectangle,
  useMap,
  useMapEvents,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

import { useStore } from '../store/useStore';
import { polygonToLatLngs, metadataToBounds } from '../utils/projection';

const FEATURE_COLORS = {
  building:   '#F97316',
  road:       '#2563EB',
  road_added: '#F59E0B',
  water:      '#06B6D4',
};

function getSubtype(det) {
  if (det.subtype) return det.subtype;
  if (det.feature_type && det.feature_type !== det.base_feature_type) return det.feature_type;
  if (det.display_label?.includes(' - ')) return det.display_label.split(' - ').slice(1).join(' - ');
  return det.feature_type;
}

function BoundsFitter({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [40, 40] });
  }, [bounds, map]);
  return null;
}

function DetectionPolygon({ det, color, isAddedRoad }) {
  const crs     = det.crs || 'EPSG:4326';
  const latLngs = useMemo(() => polygonToLatLngs(det.geo_polygon, crs), [det.geo_polygon, crs]);
  const subtype = getSubtype(det);
  const label   = det.display_label || det.feature_type || '';

  return (
    <Polygon
      positions={latLngs}
      pathOptions={{
        color,
        weight: isAddedRoad ? 3 : 1.5,
        opacity: 0.9,
        fillColor: color,
        fillOpacity: isAddedRoad ? 0.4 : 0.25,
      }}
      eventHandlers={{
        mouseover: (e) => e.target.setStyle({ fillOpacity: isAddedRoad ? 0.6 : 0.5, weight: isAddedRoad ? 4 : 2.5 }),
        mouseout:  (e) => e.target.setStyle({ fillOpacity: isAddedRoad ? 0.4 : 0.25, weight: isAddedRoad ? 3 : 1.5 }),
      }}
    >
      <Popup>
        <div className="detection-popup">
          <div
            className="popup-type-badge"
            style={{
              color,
              background: color + '22',
              border: `1px solid ${color}55`,
            }}
          >
            <svg width="8" height="8" viewBox="0 0 8 8" fill={color}>
              <circle cx="4" cy="4" r="4" />
            </svg>
            {label}
          </div>
          <div className="popup-grid">
            <span className="popup-key">Class</span>
            <span className="popup-val">{det.base_feature_type || det.feature_type}</span>
            <span className="popup-key">Type</span>
            <span className="popup-val">{subtype}</span>
            <span className="popup-key">Confidence</span>
            <span className="popup-val">{(det.confidence * 100).toFixed(1)}%</span>
            <span className="popup-key">Area</span>
            <span className="popup-val">{det.area_px?.toLocaleString()} px²</span>
            <span className="popup-key">Chunk</span>
            <span className="popup-val">{det.chunk_id}</span>
            <span className="popup-key">CRS</span>
            <span className="popup-val">{det.crs}</span>
          </div>
        </div>
      </Popup>
    </Polygon>
  );
}

export default function DetectionMap() {
  const { detections, activeFilters, imageMetadata } = useStore();

  const bounds = useMemo(
    () => (imageMetadata ? metadataToBounds(imageMetadata) : null),
    [imageMetadata]
  );

  const visibleDetections = useMemo(() => {
    return detections
      .filter(
        (d) =>
          activeFilters[d.display_label || d.feature_type] !== false &&
          d.geo_polygon?.length >= 4
      )
      .sort((a, b) => {
        if (a.source_feature_type === 'road_added' && b.source_feature_type !== 'road_added') return 1;
        if (a.source_feature_type !== 'road_added' && b.source_feature_type === 'road_added') return -1;
        return 0;
      });
  }, [detections, activeFilters]);

  const center = bounds
    ? [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    : [20.5937, 78.9629];

  return (
    <MapContainer
      center={center}
      zoom={bounds ? 14 : 5}
      maxZoom={24}
      style={{ width: '100%', height: '100%' }}
      zoomControl={true}
    >
      {/* Satellite base */}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution="Esri, Maxar, Earthstar Geographics"
        maxNativeZoom={18}
        maxZoom={24}
      />

      {/* Label overlay */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png"
        attribution="© OpenStreetMap © CartoDB"
        opacity={0.65}
        maxNativeZoom={20}
        maxZoom={24}
      />

      {/* Image bounds */}
      {bounds && (
        <Rectangle
          bounds={bounds}
          pathOptions={{
            color: '#22C55E',
            weight: 1.5,
            fill: false,
            dashArray: '8 5',
            opacity: 0.6,
          }}
        />
      )}

      <BoundsFitter bounds={bounds} />

      {/* Detections */}
      {visibleDetections.map((det, idx) => {
        const color = det.colour
          || FEATURE_COLORS[det.base_feature_type]
          || FEATURE_COLORS[det.feature_type]
          || '#FFFFFF';
        const isAddedRoad = det.source_feature_type === 'road_added';

        return (
          <DetectionPolygon
            key={`${det.chunk_id}-${idx}`}
            det={det}
            color={color}
            isAddedRoad={isAddedRoad}
          />
        );
      })}
    </MapContainer>
  );
}
