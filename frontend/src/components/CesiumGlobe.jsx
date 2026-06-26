/**
 * CesiumGlobe — 3D interactive globe with cinematic fly-in and professional navigation.
 *
 * Supports two road layer modes (roadMode: 'original' | 'enhanced'):
 *   - original: renders road detections from the SegFormer pipeline
 *   - enhanced: renders enhanced road polygons from the user-triggered pipeline
 *
 * Camera sequence on upload:
 *   1. Earth slowly rotates in global view
 *   2. Cinematic fly → India overview
 *   3. Fly → uploaded village, near top-down
 *
 * Controls: zoom in/out overlay, Google Earth-like navigation via ScreenSpaceCameraController.
 */
import React, { useEffect, useRef, useCallback } from 'react';
import { useStore } from '../store/useStore';
import { toWGS84 } from '../utils/projection';

const Cesium = window.Cesium;

function computeCameraHeight(minLng, minLat, maxLng, maxLat) {
  const dLng = Math.abs(maxLng - minLng);
  const dLat = Math.abs(maxLat - minLat);
  const latMid = (minLat + maxLat) / 2;
  const widthM  = dLng * 111320 * Math.cos((latMid * Math.PI) / 180);
  const heightM = dLat * 110574;
  const extentM = Math.max(widthM, heightM);
  return Math.max(800, Math.min(200000, extentM * 1.6));
}

export default function CesiumGlobe({ cesiumRef }) {
  const containerRef         = useRef(null);
  const viewerRef            = useRef(null);
  const entityMapRef         = useRef({ building: [], road: [], water: [] });
  const renderedCountRef     = useRef(0);
  const orbitListenerRef     = useRef(null);
  const orbitTimerRef        = useRef(null);
  const globalRotRef         = useRef(null);
  const bboxEntityRef        = useRef(null);
  const activeFiltersRef     = useRef({});

  const { detections, activeFilters, imageMetadata, jobStatus } = useStore();

  useEffect(() => { activeFiltersRef.current = activeFilters; }, [activeFilters]);

  // ── Viewer initialisation ──────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !Cesium || viewerRef.current) return;

    Cesium.Ion.defaultAccessToken =
      process.env.REACT_APP_CESIUM_TOKEN ||
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc4ZTciLCJpZCI6NTc3MzMsImlhdCI6MTYyMjY0NDE2N30.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk';

    const viewer = new Cesium.Viewer(containerRef.current, {
      terrainProvider:       new Cesium.EllipsoidTerrainProvider(),
      animation:             false,
      timeline:              false,
      geocoder:              false,
      homeButton:            false,
      sceneModePicker:       false,
      baseLayerPicker:       false,
      navigationHelpButton:  false,
      fullscreenButton:      false,
      infoBox:               true,
      selectionIndicator:    true,
    });

    viewer.cesiumWidget.showRenderLoopErrors = false;
    const creditContainer = viewer.cesiumWidget.creditContainer;
    if (creditContainer) creditContainer.style.display = 'none';

    // ESRI World Imagery — natural-looking satellite base
    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.add(
      new Cesium.ImageryLayer(
        new Cesium.UrlTemplateImageryProvider({
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          maximumLevel: 19,
          credit: 'Esri, Maxar, Earthstar Geographics',
        }),
        { brightness: 0.62, saturation: 0.88, gamma: 0.72 }
      )
    );

    // Scene atmosphere
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#070A0F');
    viewer.scene.globe.enableLighting = false;
    viewer.scene.globe.atmosphereBrightnessShift = -0.1;
    viewer.scene.globe.atmosphereSaturationShift = -0.05;
    viewer.scene.skyAtmosphere.show = true;
    viewer.scene.skyBox.show = true;
    viewer.scene.fog.enabled = true;
    viewer.scene.fog.density = 0.00008;
    viewer.scene.fog.minimumBrightness = 0.0;

    // Google Earth-like navigation feel
    const ctrl = viewer.scene.screenSpaceCameraController;
    ctrl.enableRotate    = true;
    ctrl.enableTranslate = true;
    ctrl.enableZoom      = true;
    ctrl.enableTilt      = true;
    ctrl.enableLook      = true;
    ctrl.inertiaSpin      = 0.92;
    ctrl.inertiaTranslate = 0.92;
    ctrl.inertiaZoom      = 0.82;
    ctrl.minimumZoomDistance  = 50;
    ctrl.maximumZoomDistance  = 25000000;

    // Initial: Earth perfectly centered, full globe visible, north-up
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(78.9629, 20, 15000000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-90),
        roll:    0,
      },
    });

    // Slow ambient rotation while idle (stops when upload begins)
    const globalRotFn = () => {
      if (!viewer.isDestroyed()) {
        viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.00025);
      }
    };
    viewer.scene.postRender.addEventListener(globalRotFn);
    globalRotRef.current = globalRotFn;

    viewerRef.current = viewer;

    return () => {
      if (orbitTimerRef.current) clearTimeout(orbitTimerRef.current);
      if (!viewer.isDestroyed()) {
        if (orbitListenerRef.current) {
          viewer.scene.postRender.removeEventListener(orbitListenerRef.current);
        }
        if (globalRotRef.current) {
          viewer.scene.postRender.removeEventListener(globalRotRef.current);
        }
      }
      if (!viewer.isDestroyed()) viewer.destroy();
      viewerRef.current    = null;
      entityMapRef.current = { building: [], road: [], water: [] };
      renderedCountRef.current  = 0;
    };
  }, []);

  // ── Camera controls exposed via ref ───────────────────────────────────────
  const flyToVillage = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer || !imageMetadata?.bounds) return;
    const { bounds, crs } = imageMetadata;
    const [minLng, minLat] = toWGS84(bounds.left, bounds.bottom, crs);
    const [maxLng, maxLat] = toWGS84(bounds.right, bounds.top, crs);
    const cx = (minLng + maxLng) / 2;
    const cy = (minLat + maxLat) / 2;
    const height = computeCameraHeight(minLng, minLat, maxLng, maxLat);

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(cx, cy, height),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-82),
        roll:    0,
      },
      duration: 2.5,
      easingFunction: Cesium.EasingFunction.QUINTIC_IN_OUT,
    });
  }, [imageMetadata]);

  const resetCamera = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(78.9629, 20, 15000000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-90),
        roll:    0,
      },
      duration: 2.5,
      easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
    });
  }, []);

  const zoomIn = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.camera.zoomIn(viewer.camera.positionCartographic.height * 0.35);
  }, []);

  const zoomOut = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.camera.zoomOut(viewer.camera.positionCartographic.height * 0.45);
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    document.fullscreenElement ? document.exitFullscreen?.() : el.requestFullscreen?.();
  }, []);

  useEffect(() => {
    if (cesiumRef) {
      cesiumRef.current = { flyToVillage, resetCamera, zoomIn, zoomOut, toggleFullscreen };
    }
  }, [cesiumRef, flyToVillage, resetCamera, zoomIn, zoomOut, toggleFullscreen]);

  // ── Bounding box + cinematic fly-in when metadata arrives ─────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !imageMetadata?.bounds) return;

    // Stop idle globe rotation
    if (globalRotRef.current && !viewer.isDestroyed()) {
      viewer.scene.postRender.removeEventListener(globalRotRef.current);
      globalRotRef.current = null;
    }

    const { bounds, crs } = imageMetadata;
    const [minLng, minLat] = toWGS84(bounds.left, bounds.bottom, crs);
    const [maxLng, maxLat] = toWGS84(bounds.right, bounds.top, crs);
    const cx = (minLng + maxLng) / 2;
    const cy = (minLat + maxLat) / 2;
    const finalHeight = computeCameraHeight(minLng, minLat, maxLng, maxLat);

    // Add bounding box rectangle
    if (bboxEntityRef.current && !viewer.isDestroyed()) {
      viewer.entities.remove(bboxEntityRef.current);
    }
    bboxEntityRef.current = viewer.entities.add({
      rectangle: {
        coordinates: Cesium.Rectangle.fromDegrees(minLng, minLat, maxLng, maxLat),
        material: Cesium.Color.fromCssColorString('#7BB369').withAlpha(0.06),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#7BB369').withAlpha(0.65),
        outlineWidth: 2,
        height: 0,
      },
    });

    // Stage 1: snap to global view (already there if fresh load)
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(cx, cy + 8, 14000000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-88),
        roll:    0,
      },
    });

    // Stage 2: fly toward India region
    const t1 = setTimeout(() => {
      const v = viewerRef.current;
      if (!v || v.isDestroyed()) return;
      v.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(cx, cy + 3, 900000),
        orientation: {
          heading: Cesium.Math.toRadians(0),
          pitch:   Cesium.Math.toRadians(-62),
          roll:    0,
        },
        duration: 2.6,
        easingFunction: Cesium.EasingFunction.QUINTIC_IN_OUT,
      });
    }, 180);

    // Stage 3: descend to village in top-down view
    const t2 = setTimeout(() => {
      const v = viewerRef.current;
      if (!v || v.isDestroyed()) return;
      v.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(cx, cy, finalHeight),
        orientation: {
          heading: Cesium.Math.toRadians(0),
          pitch:   Cesium.Math.toRadians(-82),
          roll:    0,
        },
        duration: 2.8,
        easingFunction: Cesium.EasingFunction.QUINTIC_IN_OUT,
      });
    }, 3100);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [imageMetadata]);

  // ── Real-time detection rendering (original roads + buildings + water) ─────
  useEffect(() => {
    if (detections.length <= renderedCountRef.current) return;

    const newDets = detections.slice(renderedCountRef.current);
    renderedCountRef.current = detections.length;

    newDets.forEach((det, idx) => {
      const { feature_type, geo_polygon, crs, confidence = 0.5 } = det;
      if (!geo_polygon || geo_polygon.length < 3) return;

      const positions = geo_polygon.map(([x, y]) => {
        const [lng, lat] = toWGS84(x, y, crs || 'EPSG:4326');
        return Cesium.Cartesian3.fromDegrees(lng, lat);
      });
      if (positions.length < 3) return;

      const delay = (idx % 30) * 20;

      setTimeout(() => {
        const v = viewerRef.current;
        if (!v || v.isDestroyed()) return;

        let entity = null;

        if (feature_type === 'building') {
          entity = v.entities.add({
            polygon: {
              hierarchy:     new Cesium.PolygonHierarchy(positions),
              material:      Cesium.Color.fromCssColorString('#CF7A3E').withAlpha(0.75),
              outline:       true,
              outlineColor:  Cesium.Color.fromCssColorString('#E89A60').withAlpha(0.9),
              outlineWidth:  2,
              extrudedHeight: 3.5 + confidence * 7,
              height:        0,
              closeTop:      true,
              closeBottom:   false,
            },
          });
        } else if (feature_type === 'road') {
          entity = v.entities.add({
            polygon: {
              hierarchy:    new Cesium.PolygonHierarchy(positions),
              material:     Cesium.Color.fromCssColorString('#4E8AB0').withAlpha(0.8),
              outline:      true,
              outlineColor: Cesium.Color.fromCssColorString('#7AACCC').withAlpha(0.95),
              outlineWidth: 2.5,
              height:       0.5,
            },
          });
        } else if (feature_type === 'water') {
          entity = v.entities.add({
            polygon: {
              hierarchy:    new Cesium.PolygonHierarchy(positions),
              material:     Cesium.Color.fromCssColorString('#3EACB0').withAlpha(0.55),
              outline:      true,
              outlineColor: Cesium.Color.fromCssColorString('#60CDD2').withAlpha(0.9),
              outlineWidth: 2,
              height:       0,
            },
          });
        }

        if (entity) {
          entity.show = activeFiltersRef.current[feature_type] !== false;
          if (entityMapRef.current[feature_type]) {
            entityMapRef.current[feature_type].push(entity);
          }
        }
      }, delay);
    });
  }, [detections]);

  // ── Layer visibility toggles ──────────────────────────────────────────────
  useEffect(() => {
    const em = entityMapRef.current;
    ['building', 'road', 'water'].forEach(type => {
      const visible = activeFilters[type] !== false;
      em[type].forEach(e => { if (e) e.show = visible; });
    });
  }, [activeFilters]);

  // ── Cinematic orbit for 5 seconds when analysis completes ─────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    if (jobStatus === 'completed') {
      const fn = () => {
        if (!viewer.isDestroyed()) viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.00016);
      };
      viewer.scene.postRender.addEventListener(fn);
      orbitListenerRef.current = fn;

      orbitTimerRef.current = setTimeout(() => {
        if (!viewer.isDestroyed() && orbitListenerRef.current) {
          viewer.scene.postRender.removeEventListener(orbitListenerRef.current);
          orbitListenerRef.current = null;
        }
      }, 5000);

      return () => {
        if (orbitTimerRef.current) clearTimeout(orbitTimerRef.current);
        if (!viewer.isDestroyed() && orbitListenerRef.current) {
          viewer.scene.postRender.removeEventListener(orbitListenerRef.current);
          orbitListenerRef.current = null;
        }
      };
    } else if (orbitListenerRef.current && !viewer.isDestroyed()) {
      viewer.scene.postRender.removeEventListener(orbitListenerRef.current);
      orbitListenerRef.current = null;
    }
  }, [jobStatus]);

  const isDone = jobStatus === 'completed';

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Floating navigation controls — upper right */}
      <div className="nav-controls">
        <button className="nav-btn" onClick={zoomIn} title="Zoom in">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <line x1="6.5" y1="1.5" x2="6.5" y2="11.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            <line x1="1.5" y1="6.5" x2="11.5" y2="6.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
        </button>
        <button className="nav-btn" onClick={zoomOut} title="Zoom out">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <line x1="1.5" y1="6.5" x2="11.5" y2="6.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
        </button>
        <div className="nav-sep" />
        <button className="nav-btn" onClick={resetCamera} title="Global view">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="2" y1="12" x2="22" y2="12"/>
            <path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/>
          </svg>
        </button>
        {imageMetadata && (
          <button className="nav-btn nav-btn--accent" onClick={flyToVillage} title="Fly to site">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="3 11 22 2 13 21 11 13 3 11"/>
            </svg>
          </button>
        )}
        <div className="nav-sep" />
        <button className="nav-btn" onClick={toggleFullscreen} title="Fullscreen">
          <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
            <path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      {/* Empty state — shown before any upload */}
      {!imageMetadata && (
        <div className="globe-empty">
          <div className="globe-empty-inner">
            <div className="globe-empty-icon">
              <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="2" y1="12" x2="22" y2="12"/>
                <path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20"/>
              </svg>
            </div>
            <div className="globe-empty-title">Upload a GeoTIFF to begin</div>
            <div className="globe-empty-sub">Satellite orthophoto analysis · SegFormer AI</div>
          </div>
        </div>
      )}

      {/* Analysis complete banner */}
      {isDone && (
        <div className="analysis-complete-badge">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          <span>Analysis Complete</span>
        </div>
      )}
    </div>
  );
}
