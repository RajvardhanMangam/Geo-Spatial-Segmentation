/**
 * CesiumGlobe — 3D Earth visualization using CesiumJS (loaded via CDN).
 *
 * Behaviour summary:
 *  – Slow idle rotation while waiting for the user to start analysis
 *  – When "Start Analysis" is clicked (jobStatus → queued) the camera flies
 *    to the uploaded GeoTIFF bounds, rotation stops, and an animated green
 *    bounding box is drawn edge-by-edge after landing
 *  – Detection polygons stream in and are rendered incrementally on the globe
 *  – Header nav buttons control the camera via globeAPI (populated here)
 *  – Filter visibility toggled via Zustand activeFilters
 */
import React, { useEffect, useRef, useState } from 'react';
import { useStore } from '../store/useStore';
import { metadataToBounds, polygonToLatLngs } from '../utils/projection';
import { globeAPI } from '../utils/globeRef';

/* ── Feature colours (RGBA 0-1) ─────────────────────────── */
const COLORS = {
  building:   [249 / 255, 115 / 255, 22  / 255, 0.50],
  road:       [37  / 255, 99  / 255, 235 / 255, 0.55],
  road_added: [245 / 255, 158 / 255, 11  / 255, 0.55],
  water:      [6   / 255, 182 / 255, 212 / 255, 0.45],
};

function cesiumColor(det) {
  const Cesium = window.Cesium;
  if (det.colour) {
    try { return Cesium.Color.fromCssColorString(det.colour).withAlpha(0.5); } catch (_) {}
  }
  const c = COLORS[det.base_feature_type] || COLORS[det.feature_type] || [1, 1, 1, 0.35];
  return new Cesium.Color(c[0], c[1], c[2], c[3]);
}

/* ── Loading overlay ─────────────────────────────────────── */
function GlobeLoader({ ready }) {
  if (ready) return null;
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#0B1220', zIndex: 5, flexDirection: 'column', gap: 12,
    }}>
      <div style={{ width: 56, height: 56 }}>
        <svg viewBox="0 0 56 56" style={{ width: 56, height: 56, animation: 'spin 3s linear infinite' }}>
          <circle cx="28" cy="28" r="24" fill="none" stroke="rgba(34,197,94,0.15)" strokeWidth="3" />
          <circle cx="28" cy="28" r="24" fill="none" stroke="#22C55E" strokeWidth="3"
            strokeDasharray="40 110" strokeLinecap="round" />
        </svg>
      </div>
      <span style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 12, color: 'rgba(156,163,175,0.8)', letterSpacing: 1 }}>
        Initialising 3D Globe…
      </span>
      <style>{`@keyframes spin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }`}</style>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */
export default function CesiumGlobe() {
  const containerRef = useRef(null);
  const viewerRef    = useRef(null);
  const dsRef        = useRef(null);    // CustomDataSource for detection polygons
  const bbRef        = useRef([]);      // array of bounding-box entities
  const rotRef       = useRef(true);    // idle rotation flag
  const orbitRef     = useRef(false);   // post-completion orbit flag
  const tickRef      = useRef(null);    // onTick removal fn
  const addedRef     = useRef(0);       // polygons already pushed to Cesium
  const flyDoneRef   = useRef(false);   // fly-to happened this session
  const sessionRef   = useRef(0);       // incremented on reset to cancel stale timeouts
  const mountedRef   = useRef(true);
  const [cesiumReady, setCesiumReady] = useState(false);

  const { detections, activeFilters, imageMetadata, jobStatus, jobId, uploadStatus } = useStore();

  /* ── 1. Initialise Cesium Viewer & populate globeAPI ─────── */
  useEffect(() => {
    mountedRef.current = true;

    const init = async () => {
      let attempts = 0;
      while (!window.Cesium && attempts < 100) {
        await new Promise((r) => setTimeout(r, 100));
        attempts++;
      }
      if (!window.Cesium || !mountedRef.current) return;

      const Cesium = window.Cesium;
      Cesium.Ion.defaultAccessToken = process.env.REACT_APP_CESIUM_TOKEN || '';

      let viewer;
      try {
        viewer = new Cesium.Viewer(containerRef.current, {
          imageryProvider:       false,
          terrainProvider:       new Cesium.EllipsoidTerrainProvider(),
          baseLayerPicker:       false,
          geocoder:              false,
          homeButton:            false,
          sceneModePicker:       false,
          navigationHelpButton:  false,
          animation:             false,
          timeline:              false,
          fullscreenButton:      false,
          vrButton:              false,
          infoBox:               false,
          selectionIndicator:    false,
          requestRenderMode:     false,
          maximumRenderTimeChange: Infinity,
        });
      } catch (err) {
        console.error('[Cesium] Viewer init failed:', err);
        return;
      }

      if (!mountedRef.current) { viewer.destroy(); return; }

      /* ESRI World Imagery (no ion token needed) */
      viewer.imageryLayers.removeAll();
      try {
        if (typeof Cesium.ArcGisMapServerImageryProvider.fromUrl === 'function') {
          const esri = await Cesium.ArcGisMapServerImageryProvider.fromUrl(
            'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer',
            { enablePickFeatures: false }
          );
          if (mountedRef.current) viewer.imageryLayers.addImageryProvider(esri);
        } else {
          viewer.imageryLayers.addImageryProvider(
            new Cesium.ArcGisMapServerImageryProvider({
              url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer',
            })
          );
        }
      } catch (e) {
        console.warn('[Cesium] ESRI imagery failed, globe may appear dark:', e);
      }

      if (!mountedRef.current) { viewer.destroy(); return; }

      viewer.scene.globe.enableLighting    = true;
      viewer.scene.atmosphere.show         = true;
      viewer.scene.skyAtmosphere.show      = true;
      viewer.scene.fog.enabled             = true;
      viewer.scene.fog.density             = 0.00008;
      viewer.scene.backgroundColor         = Cesium.Color.fromCssColorString('#0B1220');

      /* Initial camera — above India */
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(82, 22, 9_000_000),
        orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 },
      });

      /* Detection data source */
      const ds = new Cesium.CustomDataSource('detections');
      await viewer.dataSources.add(ds);
      dsRef.current = ds;

      /* Idle rotation / orbit tick */
      tickRef.current = viewer.clock.onTick.addEventListener(() => {
        if (orbitRef.current) {
          viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.00004);
        } else if (rotRef.current) {
          viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.000055);
        }
      });

      viewerRef.current = viewer;

      /* ── Populate globeAPI so Header buttons work ── */
      globeAPI.viewer   = viewer;
      globeAPI.rotRef   = rotRef;
      globeAPI.orbitRef = orbitRef;

      globeAPI.flyToGlobe = () => {
        if (!viewer || viewer.isDestroyed()) return;
        rotRef.current   = true;
        orbitRef.current = false;
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(82, 22, 9_000_000),
          orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 },
          duration: 2.5,
          easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
        });
      };

      globeAPI.flyToArea = () => {
        if (!viewer || viewer.isDestroyed()) return;

        // Use cached bounds, or compute on the fly from stored imageMetadata
        let b = globeAPI.bounds;
        if (!b && globeAPI.imageMetadata) {
          const computed = metadataToBounds(globeAPI.imageMetadata);
          if (computed) {
            const [[minLat, minLng], [maxLat, maxLng]] = computed;
            b = { minLng, minLat, maxLng, maxLat };
            globeAPI.bounds = b;
          }
        }
        if (!b) return;

        const { minLng, minLat, maxLng, maxLat } = b;
        rotRef.current   = false;
        orbitRef.current = false;
        viewer.camera.flyTo({
          destination: Cesium.Rectangle.fromDegrees(
            minLng - 0.005, minLat - 0.005,
            maxLng + 0.005, maxLat + 0.005
          ),
          duration: 1.8,
          easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
        });
      };

      globeAPI.fitBounds = () => {
        if (!viewer || viewer.isDestroyed()) return;
        let b = globeAPI.bounds;
        if (!b && globeAPI.imageMetadata) {
          const computed = metadataToBounds(globeAPI.imageMetadata);
          if (computed) {
            const [[minLat, minLng], [maxLat, maxLng]] = computed;
            b = { minLng, minLat, maxLng, maxLat };
            globeAPI.bounds = b;
          }
        }
        if (!b) return;
        const { minLng, minLat, maxLng, maxLat } = b;
        rotRef.current   = false;
        orbitRef.current = false;
        viewer.camera.flyTo({
          destination: Cesium.Rectangle.fromDegrees(minLng, minLat, maxLng, maxLat),
          duration: 1.5,
          easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
        });
      };

      globeAPI.resetCamera = () => {
        if (!viewer || viewer.isDestroyed()) return;
        const pos = viewer.camera.positionCartographic;
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(
            Cesium.Math.toDegrees(pos.longitude),
            Cesium.Math.toDegrees(pos.latitude),
            pos.height
          ),
          orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 },
          duration: 1.2,
          easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
        });
      };

      setCesiumReady(true);
    };

    init();

    return () => {
      mountedRef.current = false;
      if (tickRef.current) { tickRef.current(); tickRef.current = null; }
      if (viewerRef.current && !viewerRef.current.isDestroyed()) viewerRef.current.destroy();
      viewerRef.current = null;
      dsRef.current     = null;
      globeAPI.viewer        = null;
      globeAPI.bounds        = null;
      globeAPI.imageMetadata = null;
      globeAPI.flyToGlobe    = () => {};
      globeAPI.flyToArea     = () => {};
      globeAPI.fitBounds     = () => {};
      globeAPI.resetCamera   = () => {};
      setCesiumReady(false);
    };
  }, []);

  /* ── 2. Reset when store clears ──────────────────────────── */
  useEffect(() => {
    if (uploadStatus !== 'idle') return;

    sessionRef.current++; // invalidate any in-flight timeouts

    if (dsRef.current) { dsRef.current.entities.removeAll(); addedRef.current = 0; }

    if (bbRef.current.length && viewerRef.current && !viewerRef.current.isDestroyed()) {
      bbRef.current.forEach((e) => viewerRef.current.entities.remove(e));
    }
    bbRef.current = [];

    rotRef.current         = true;
    orbitRef.current       = false;
    flyDoneRef.current     = false;
    globeAPI.bounds        = null;
    globeAPI.imageMetadata = null;
  }, [uploadStatus]);

  /* ── 3. Cache bounds when image metadata arrives ─────────── */
  useEffect(() => {
    if (!imageMetadata) { globeAPI.imageMetadata = null; return; }
    globeAPI.imageMetadata = imageMetadata; // store raw metadata for flyToArea fallback
    const bounds = metadataToBounds(imageMetadata);
    if (!bounds) return;
    const [[minLat, minLng], [maxLat, maxLng]] = bounds;
    globeAPI.bounds = { minLng, minLat, maxLng, maxLat };
  }, [imageMetadata]);

  /* ── 4. Fly to area + animated bounding box on job start ─── */
  useEffect(() => {
    // Trigger on jobId becoming non-null (job started) OR jobStatus entering active states.
    // Watching jobId is more reliable — it fires exactly once when inference begins.
    const ACTIVE_STATUSES = ['queued', 'started', 'running'];
    const jobIsActive = !!jobId || ACTIVE_STATUSES.includes(jobStatus);
    if (!jobIsActive || !imageMetadata) return;
    if (!viewerRef.current || !cesiumReady || viewerRef.current.isDestroyed()) return;
    if (flyDoneRef.current) return;

    const Cesium  = window.Cesium;
    const viewer  = viewerRef.current;
    const session = sessionRef.current;

    // Compute bounds first — only mark fly as done if we can actually fly
    const bounds = metadataToBounds(imageMetadata);
    if (!bounds) {
      console.warn('[GeoSight] metadataToBounds returned null — cannot fly to area', imageMetadata);
      return;
    }
    flyDoneRef.current = true; // mark AFTER confirming bounds are valid

    const [[minLat, minLng], [maxLat, maxLng]] = bounds;

    // Also ensure globeAPI.bounds is set (in case Effect 3 missed it)
    globeAPI.bounds = { minLng, minLat, maxLng, maxLat };

    rotRef.current   = false;
    orbitRef.current = false;

    /* Clear any existing bounding box */
    if (bbRef.current.length) {
      bbRef.current.forEach((e) => viewer.entities.remove(e));
    }
    const bbEntities = [];
    bbRef.current = bbEntities; // shared reference — timeouts push into this array

    /* ── Smooth fly-to (2.8 s) — use Rectangle so Cesium sizes the view correctly ── */
    const FLY_MS = 2800;
    const pad    = Math.max((maxLng - minLng) * 0.08, 0.002);  // 8% padding, min 200m

    viewer.camera.flyTo({
      destination: Cesium.Rectangle.fromDegrees(
        minLng - pad, minLat - pad,
        maxLng + pad, maxLat + pad
      ),
      duration: FLY_MS / 1000,
      easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
    });

    console.info('[GeoSight] Flying to', { minLng, minLat, maxLng, maxLat, pad });

    /* ── Semi-transparent fill rectangle (fades in over 0.8 s after landing) ── */
    const fillStart = Date.now() + FLY_MS;
    const fillEntity = viewer.entities.add({
      rectangle: {
        coordinates: Cesium.Rectangle.fromDegrees(minLng, minLat, maxLng, maxLat),
        material: new Cesium.ColorMaterialProperty(
          new Cesium.CallbackProperty(() => {
            const elapsed = Math.max(0, Date.now() - fillStart) / 800;
            const alpha   = Math.min(0.12, elapsed * 0.12);
            return new Cesium.Color(34 / 255, 197 / 255, 94 / 255, alpha);
          }, false)
        ),
        outline: false,
        height: 0,
      },
    });
    bbEntities.push(fillEntity);

    /* ── Animated edge draw: reveal each edge with a 280 ms stagger ── */
    const edgePositions = [
      // bottom
      [Cesium.Cartesian3.fromDegrees(minLng, minLat, 0), Cesium.Cartesian3.fromDegrees(maxLng, minLat, 0)],
      // right
      [Cesium.Cartesian3.fromDegrees(maxLng, minLat, 0), Cesium.Cartesian3.fromDegrees(maxLng, maxLat, 0)],
      // top
      [Cesium.Cartesian3.fromDegrees(maxLng, maxLat, 0), Cesium.Cartesian3.fromDegrees(minLng, maxLat, 0)],
      // left
      [Cesium.Cartesian3.fromDegrees(minLng, maxLat, 0), Cesium.Cartesian3.fromDegrees(minLng, minLat, 0)],
    ];

    edgePositions.forEach((positions, i) => {
      setTimeout(() => {
        if (sessionRef.current !== session) return;
        if (!viewerRef.current || viewerRef.current.isDestroyed()) return;
        const edge = viewer.entities.add({
          polyline: {
            positions,
            width: 2.5,
            material: new Cesium.PolylineGlowMaterialProperty({
              glowPower: 0.3,
              color: Cesium.Color.fromCssColorString('#22C55E').withAlpha(0.9),
            }),
            clampToGround: true,
          },
        });
        bbEntities.push(edge);
      }, FLY_MS + i * 280);
    });

    /* ── Corner markers after all edges drawn ── */
    const corners = [
      [minLng, minLat], [maxLng, minLat],
      [maxLng, maxLat], [minLng, maxLat],
    ];

    setTimeout(() => {
      if (sessionRef.current !== session) return;
      if (!viewerRef.current || viewerRef.current.isDestroyed()) return;
      corners.forEach(([lng, lat]) => {
        const pt = viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(lng, lat, 0),
          point: {
            pixelSize: 9,
            color: Cesium.Color.fromCssColorString('#22C55E'),
            outlineColor: Cesium.Color.WHITE.withAlpha(0.6),
            outlineWidth: 1.5,
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          },
        });
        bbEntities.push(pt);
      });
    }, FLY_MS + edgePositions.length * 280 + 100);
  }, [jobId, jobStatus, imageMetadata, cesiumReady]);

  /* ── 5. Stop orbit when job completes (let user navigate) ── */
  useEffect(() => {
    if (jobStatus !== 'completed') return;
    orbitRef.current = false;
  }, [jobStatus]);

  /* ── 6. Add detection polygons incrementally ─────────────── */
  useEffect(() => {
    if (!dsRef.current || !cesiumReady || !window.Cesium) return;
    const Cesium   = window.Cesium;
    const entities = dsRef.current.entities;

    if (detections.length === 0) {
      if (addedRef.current > 0) { entities.removeAll(); addedRef.current = 0; }
      return;
    }

    const newDets = detections.slice(addedRef.current);
    if (newDets.length === 0) return;

    entities.suspendEvents();
    newDets.forEach((det, i) => {
      try {
        const crs     = det.crs || 'EPSG:4326';
        const latLngs = polygonToLatLngs(det.geo_polygon, crs);
        if (latLngs.length < 3) return;

        const positions = latLngs.map(([lat, lng]) =>
          Cesium.Cartesian3.fromDegrees(lng, lat, 0)
        );
        const color   = cesiumColor(det);
        const label   = det.display_label || det.feature_type;
        const visible = activeFilters[label] !== false;

        entities.add({
          id:   `det-${addedRef.current + i}`,
          name: label,
          polygon: {
            hierarchy:         new Cesium.PolygonHierarchy(positions),
            material:          new Cesium.ColorMaterialProperty(color),
            outline:           true,
            outlineColor:      color.brighten(0.35, new Cesium.Color()),
            outlineWidth:      1.5,
            height:            0,
            perPositionHeight: false,
          },
          show: visible,
        });
      } catch (_) { /* skip malformed polygon */ }
    });
    entities.resumeEvents();

    addedRef.current = detections.length;
  }, [detections, cesiumReady]);

  /* ── 7. Toggle filter visibility ────────────────────────── */
  useEffect(() => {
    if (!dsRef.current || !cesiumReady) return;
    const values = dsRef.current.entities.values;
    for (let i = 0; i < values.length; i++) {
      const e = values[i];
      if (e.name) e.show = activeFilters[e.name] !== false;
    }
  }, [activeFilters, cesiumReady]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <GlobeLoader ready={cesiumReady} />
    </div>
  );
}
