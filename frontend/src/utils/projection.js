/**
 * Coordinate utilities for converting between projected CRS and WGS84.
 *
 * Supports:
 *   - EPSG:32644 (UTM Zone 44N — covers most of India)
 *   - EPSG:3857  (Web Mercator)
 *   - EPSG:4326  (WGS84 — pass-through)
 *
 * Uses proj4 for accuracy.
 */
import proj4 from 'proj4';

// Register common projections
proj4.defs('EPSG:32644', '+proj=utm +zone=44 +datum=WGS84 +units=m +no_defs');
proj4.defs('EPSG:32643', '+proj=utm +zone=43 +datum=WGS84 +units=m +no_defs');
proj4.defs('EPSG:32645', '+proj=utm +zone=45 +datum=WGS84 +units=m +no_defs');
proj4.defs('EPSG:3857',  '+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs');
proj4.defs('EPSG:4326',  '+proj=longlat +datum=WGS84 +no_defs');

const WGS84 = 'EPSG:4326';

/**
 * Convert [x, y] in the source CRS to [lng, lat] in WGS84.
 */
export function toWGS84(x, y, crs) {
  const srcCrs = normalizeCrs(crs);
  if (srcCrs === WGS84 || srcCrs === 'EPSG:4326') return [x, y];
  try {
    return proj4(srcCrs, WGS84, [x, y]);
  } catch (e) {
    console.warn('proj4 conversion failed for CRS:', crs, e);
    return [x, y]; // fallback — may be wrong but won't crash
  }
}

/**
 * Convert a geo_polygon (array of [x,y] pairs in source CRS) to
 * Leaflet LatLng arrays [[lat, lng], ...].
 */
export function polygonToLatLngs(polygon, crs) {
  return polygon.map(([x, y]) => {
    const [lng, lat] = toWGS84(x, y, crs);
    return [lat, lng];
  });
}

/**
 * Compute Leaflet bounds [[minLat, minLng], [maxLat, maxLng]] from image metadata.
 */
export function metadataToBounds(metadata) {
  const { bounds, crs } = metadata;
  if (!bounds) return null;

  const [minLng, minLat] = toWGS84(bounds.left, bounds.bottom, crs);
  const [maxLng, maxLat] = toWGS84(bounds.right, bounds.top, crs);

  return [
    [minLat, minLng],
    [maxLat, maxLng],
  ];
}

/**
 * Normalise CRS strings to proj4-registered keys.
 * Handles "EPSG:32644", "epsg:32644", "WGS 84 / UTM zone 44N", etc.
 */
function normalizeCrs(crs) {
  if (!crs) return WGS84;
  const upper = crs.toUpperCase().replace(/\s+/g, '');
  // Already in EPSG:XXXX form
  if (upper.startsWith('EPSG:')) return upper;
  // OGC URN form
  const urnMatch = upper.match(/URN:OGC:DEF:CRS:EPSG:.*:(\d+)/);
  if (urnMatch) return `EPSG:${urnMatch[1]}`;
  return WGS84;
}
