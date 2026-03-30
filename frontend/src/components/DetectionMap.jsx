/**
 * DetectionMap — full-screen Leaflet map with:
 *  - Satellite/OSM base tiles
 *  - Real-time GeoJSON detection overlays streamed from WebSocket
 *  - Image bounds overlay (shows processed area)
 *  - Per-feature-type colour coding
 *  - Click-to-inspect popups
 */
import React, { useEffect, useRef, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  Polygon,
  Popup,
  Rectangle,
  useMap,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

import { useStore } from '../store/useStore';
import { polygonToLatLngs, metadataToBounds } from '../utils/projection';

const FEATURE_COLORS = {
  building:   '#FF4444',
  road:       '#4488FF',
  utility:    '#FFAA00',
  vegetation: '#44BB44',
  water:      '#00BBFF',
};

// Fit map to image bounds when metadata arrives
function BoundsFitter({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [bounds, map]);
  return null;
}

export default function DetectionMap() {
  const { detections, activeFilters, imageMetadata } = useStore();

  const bounds = useMemo(
    () => (imageMetadata ? metadataToBounds(imageMetadata) : null),
    [imageMetadata]
  );

  // Convert detections to Leaflet polygons, filtered by type and visibility
  const visibleDetections = useMemo(() => {
    return detections.filter(
      (d) => activeFilters[d.feature_type] !== false && d.geo_polygon?.length >= 4
    );
  }, [detections, activeFilters]);

  const center = bounds
    ? [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    : [20.5937, 78.9629]; // India centre fallback

  return (
    <MapContainer
      center={center}
      zoom={bounds ? 14 : 5}
      style={{ width: '100%', height: '100%' }}
      zoomControl={true}
    >
      {/* Satellite base layer */}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution="Esri, Maxar, Earthstar Geographics"
        maxZoom={22}
      />

      {/* OSM labels overlay */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png"
        attribution="© OpenStreetMap, © CartoDB"
        opacity={0.7}
      />

      {/* Image bounds rectangle */}
      {bounds && (
        <Rectangle
          bounds={bounds}
          pathOptions={{
            color: '#00FF88',
            weight: 2,
            fill: false,
            dashArray: '6 4',
            opacity: 0.8,
          }}
        />
      )}

      {/* Fit map when bounds are ready */}
      <BoundsFitter bounds={bounds} />

      {/* Detection polygons */}
      {visibleDetections.map((det, idx) => {
        const crs = det.crs || 'EPSG:4326';
        const latLngs = polygonToLatLngs(det.geo_polygon, crs);
        const color = FEATURE_COLORS[det.feature_type] || '#FFFFFF';

        return (
          <Polygon
            key={`${det.chunk_id}-${idx}`}
            positions={latLngs}
            pathOptions={{
              color,
              weight: 1.5,
              opacity: 0.9,
              fillColor: color,
              fillOpacity: 0.25,
            }}
          >
            <Popup>
              <div style={{ fontFamily: 'monospace', fontSize: 13 }}>
                <strong style={{ color, textTransform: 'uppercase' }}>
                  {det.feature_type}
                </strong>
                <br />
                Confidence: {(det.confidence * 100).toFixed(1)}%
                <br />
                Area: {det.area_px?.toLocaleString()} px²
                <br />
                Chunk: {det.chunk_id}
                <br />
                CRS: {det.crs}
              </div>
            </Popup>
          </Polygon>
        );
      })}
    </MapContainer>
  );
}
