import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { metadataToBounds } from '../utils/projection';

const MAP_STYLE = {
  version: 8,
  sources: {
    satellite: {
      type: 'raster',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      attribution: '© Esri, Maxar',
      maxzoom: 19,
    },
  },
  layers: [
    { id: 'bg',  type: 'background', paint: { 'background-color': '#010203' } },
    { id: 'sat', type: 'raster',     source: 'satellite' },
  ],
};

const SRC  = 'bbox';
const LINE = 'bbox-line';
const GLOW = 'bbox-glow';
const EMPTY = { type: 'FeatureCollection', features: [] };

// ── Equirectangular canvas Earth texture ───────────────────────────────────
function makeEarthCanvas() {
  const W = 2048, H = 1024;
  const cv  = document.createElement('canvas');
  cv.width  = W;
  cv.height = H;
  const ctx = cv.getContext('2d');

  // lon/lat → pixel
  const px = (lon, lat) => [(lon + 180) / 360 * W, (90 - lat) / 180 * H];

  function poly(pts, fill, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    pts.slice(1).forEach(p => ctx.lineTo(...p));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // ── Deep ocean ──────────────────────────────────────────────────────────
  ctx.fillStyle = '#0c1e3c';
  ctx.fillRect(0, 0, W, H);

  // Subtle ocean gradient
  const og = ctx.createLinearGradient(0, 0, 0, H);
  og.addColorStop(0,   'rgba(0,10,40,0.55)');
  og.addColorStop(0.5, 'rgba(5,20,70,0.25)');
  og.addColorStop(1,   'rgba(0,10,40,0.55)');
  ctx.fillStyle = og;
  ctx.fillRect(0, 0, W, H);

  // ── North America ──────────────────────────────────────────────────────
  poly([
    px(-168,72), px(-140,73), px(-95,74), px(-60,68),
    px(-52,47),  px(-66,45),  px(-70,42), px(-75,35),
    px(-81,31),  px(-80,25),  px(-90,29), px(-97,26),
    px(-105,20), px(-88,15),  px(-77,8),  px(-78,11),
    px(-105,22), px(-118,32), px(-120,35), px(-124,48),
    px(-130,58), px(-140,60), px(-168,72),
  ], '#3a6e2a');

  // US/Canadian forest and grassland interior
  poly([
    px(-124,48), px(-75,43),  px(-75,35), px(-90,29),
    px(-97,26),  px(-120,22), px(-120,35), px(-124,48),
  ], '#4e8035');

  // Mexico arid
  poly([
    px(-118,32), px(-96,18), px(-88,15), px(-105,20),
    px(-118,28), px(-118,32),
  ], '#8a8f38');

  // Greenland ice sheet
  poly([
    px(-72,83), px(-25,82), px(-17,77), px(-22,70),
    px(-44,60), px(-52,65), px(-62,66), px(-72,76),
  ], '#c5dcea');

  // ── South America ──────────────────────────────────────────────────────
  poly([
    px(-78,10), px(-60,12), px(-50,5),  px(-34,-5),
    px(-34,-10),px(-37,-20),px(-45,-28),px(-52,-34),
    px(-68,-55),px(-76,-50),px(-80,-34),px(-80,-20),
    px(-76,0),  px(-78,5),  px(-78,10),
  ], '#2a6222');

  // Amazon basin (dark rainforest)
  poly([
    px(-78,3),  px(-50,5),  px(-45,-2), px(-55,-12),
    px(-72,-10),px(-78,3),
  ], '#124e10');

  // Cerrado savanna
  poly([
    px(-50,-5), px(-36,-10), px(-37,-20), px(-52,-20),
    px(-60,-15),px(-50,-5),
  ], '#7a9230');

  // ── Europe ────────────────────────────────────────────────────────────
  poly([
    px(-10,60), px(-5,54),  px(0,50),   px(10,48),
    px(20,50),  px(28,57),  px(30,65),  px(25,70),
    px(12,70),  px(5,62),   px(-5,56),  px(-10,60),
  ], '#4a7835');

  poly([px(-10,44), px(3,44),  px(3,36), px(-9,36),  px(-10,44)], '#608535'); // Iberia
  poly([px(6,44),   px(12,47), px(18,40),px(15,37),  px(9,40),   px(6,44)],  '#4a7835'); // Italy

  // Scandinavia
  poly([
    px(5,62),  px(15,70), px(28,71), px(30,65),
    px(20,57), px(10,57), px(5,62),
  ], '#3e6e30');

  // ── Africa ───────────────────────────────────────────────────────────
  // Sahara (desert tan)
  poly([
    px(-18,37), px(37,30), px(43,20), px(32,15),
    px(20,15),  px(0,15),  px(-18,20),px(-18,37),
  ], '#c0982a');

  // Sub-Saharan / forest belt
  poly([
    px(-18,20), px(0,15),  px(15,12), px(45,5),
    px(52,5),   px(44,-12),px(35,-5), px(20,0),
    px(5,0),    px(-18,5), px(-18,20),
  ], '#3a7025');

  // Congo rainforest
  poly([
    px(10,5),  px(30,5),  px(30,-5),
    px(20,-10),px(10,-5), px(10,5),
  ], '#0e4a0e');

  // Southern Africa
  poly([
    px(15,-20),px(35,-20),px(40,-25),
    px(32,-35),px(18,-35),px(15,-20),
  ], '#5a7e28');

  // East African highlands
  poly([
    px(35,5),px(50,12),px(52,5),px(44,-5),px(38,0),px(35,5),
  ], '#8a7a30');

  // North African coast (Mediterranean green)
  poly([
    px(-5,37),px(37,30),px(37,34),px(10,37),px(-5,37),
  ], '#6a8530', 0.5);

  // ── Middle East / Arabia ──────────────────────────────────────────────
  poly([
    px(36,37), px(60,22), px(60,12),
    px(45,12), px(43,20), px(37,30), px(36,37),
  ], '#c49030');

  // ── Asia ─────────────────────────────────────────────────────────────
  // Main landmass
  poly([
    px(26,72),  px(70,72),  px(140,72), px(170,65),
    px(160,55), px(150,45), px(130,35), px(120,22),
    px(100,20), px(78,35),  px(60,22),  px(36,37),
    px(28,57),  px(26,72),
  ], '#48723a');

  // Siberian tundra (gray-green)
  poly([
    px(60,72),  px(170,65), px(175,72),
    px(120,75), px(60,72),
  ], '#5a6848');

  // Central Asian steppe/desert
  poly([
    px(45,45),  px(90,45),  px(90,30),
    px(60,28),  px(45,38),  px(45,45),
  ], '#a09035');

  // Gobi desert
  poly([
    px(90,45), px(120,45), px(120,35),
    px(90,35), px(90,45),
  ], '#b0a040');

  // Indian subcontinent
  poly([
    px(68,37),  px(79,36),  px(88,22), px(80,8),
    px(72,8),   px(68,20),  px(68,37),
  ], '#58802e');

  // Southeast Asia
  poly([
    px(100,22), px(120,22), px(125,10),
    px(115,5),  px(100,5),  px(100,22),
  ], '#357025');

  // China coast (greener)
  poly([
    px(110,25), px(126,25), px(130,35),
    px(120,40), px(110,35), px(110,25),
  ], '#3e7028');

  // Japan
  poly([
    px(130,32), px(140,32), px(142,39),
    px(140,44), px(130,38), px(130,32),
  ], '#3e7028');

  // Indonesia
  poly([px(95,-5),  px(118,-5),  px(118,-8), px(95,-8),  px(95,-5)],  '#357025');
  poly([px(120,-2), px(136,-2),  px(136,-8), px(120,-8), px(120,-2)], '#357025');

  // ── Australia ────────────────────────────────────────────────────────
  // Arid interior
  poly([
    px(114,-22), px(130,-13), px(145,-16), px(154,-28),
    px(150,-38), px(138,-38), px(130,-34), px(114,-28),
  ], '#bf7a28');

  // Eastern coastal forest
  poly([
    px(145,-15), px(154,-28), px(153,-37),
    px(148,-38), px(147,-28), px(145,-20),
  ], '#3e7025');

  // SW Australia (Mediterranean)
  poly([
    px(114,-22), px(120,-17), px(130,-14),
    px(130,-22), px(120,-26), px(114,-22),
  ], '#50782e');

  // New Zealand
  poly([px(166,-44), px(172,-40), px(172,-35), px(166,-40)], '#3a7025');
  poly([px(172,-36), px(178,-36), px(178,-34), px(172,-35)], '#3a7025');

  // ── Antarctic ice ─────────────────────────────────────────────────────
  {
    const g = ctx.createLinearGradient(0, H - 75, 0, H);
    g.addColorStop(0, 'rgba(218,232,248,0)');
    g.addColorStop(1, 'rgba(218,232,248,1)');
    ctx.fillStyle = g;
    ctx.fillRect(0, H - 75, W, 75);
  }

  // ── Arctic ice ────────────────────────────────────────────────────────
  {
    const g = ctx.createLinearGradient(0, 0, 0, 85);
    g.addColorStop(0, 'rgba(225,238,250,1)');
    g.addColorStop(1, 'rgba(225,238,250,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, 85);
  }

  // ── Cloud wisps ──────────────────────────────────────────────────────
  ctx.save();
  for (let i = 0; i < 30; i++) {
    const cx = Math.random() * W;
    const cy = H * 0.1 + Math.random() * H * 0.8;
    const rx = 50 + Math.random() * 200;
    const ry = 10 + Math.random() * 30;
    const g  = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx);
    g.addColorStop(0, 'rgba(255,255,255,0.38)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, Math.random() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  return cv;
}

// ── Build Three.js scene ────────────────────────────────────────────────────
function buildThreeScene(container) {
  const W = container.clientWidth;
  const H = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 500);
  camera.position.z = 2.7;

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.22));
  const sun = new THREE.DirectionalLight(0xffffff, 1.15);
  sun.position.set(5, 3, 5);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x3355aa, 0.12);
  fill.position.set(-5, -2, -4);
  scene.add(fill);

  // Stars
  const starPos = new Float32Array(5000 * 3);
  for (let i = 0; i < starPos.length; i++) starPos[i] = (Math.random() - 0.5) * 200;
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
    color: 0xffffff, size: 0.09, transparent: true, opacity: 0.8,
  })));

  // Outer atmosphere glow (back-face, additive)
  const outerAtmosMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float i = pow(max(0.0, 0.58 - dot(vNormal, vec3(0,0,1))), 3.5);
        gl_FragColor = vec4(0.10, 0.32, 0.96, 1.0) * i;
      }
    `,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.14, 64, 64), outerAtmosMat));

  // Earth sphere
  const earthMat = new THREE.MeshPhongMaterial({
    map:       new THREE.CanvasTexture(makeEarthCanvas()),
    specular:  new THREE.Color(0x112244),
    shininess: 12,
  });

  // Try loading real NASA/Three texture (non-blocking)
  const loader = new THREE.TextureLoader();
  loader.crossOrigin = 'anonymous';
  const tryLoad = (i, urls) => {
    if (i >= urls.length) return;
    loader.load(
      urls[i],
      (t) => { earthMat.map = t; earthMat.needsUpdate = true; },
      undefined,
      () => tryLoad(i + 1, urls),
    );
  };
  tryLoad(0, [
    'https://unpkg.com/three@0.167.0/examples/textures/land_ocean_ice_cloud_2048.jpg',
    'https://cdn.jsdelivr.net/npm/three@0.167.0/examples/textures/land_ocean_ice_cloud_2048.jpg',
  ]);

  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), earthMat);
  scene.add(earth);

  // Inner atmosphere rim (front-face, additive)
  const innerAtmosMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float i = pow(max(0.0, 0.60 - dot(vNormal, vec3(0,0,1))), 5.0);
        gl_FragColor = vec4(0.22, 0.56, 1.0, 1.0) * i;
      }
    `,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.005, 64, 64), innerAtmosMat));

  // Resize
  const onResize = () => {
    const nw = container.clientWidth, nh = container.clientHeight;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  };
  window.addEventListener('resize', onResize);

  let animId;
  const animate = () => {
    animId = requestAnimationFrame(animate);
    earth.rotation.y += 0.0017;
    renderer.render(scene, camera);
  };
  animate();

  return () => {
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', onResize);
    renderer.dispose();
    if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
  };
}

// ── Component ───────────────────────────────────────────────────────────────
export default function GlobeView({ imageMetadata, jobStatus, viewMode }) {
  const threeRef  = useRef(null);
  const mapRef    = useRef(null);
  const mapInst   = useRef(null);
  const mapReady  = useRef(false);
  const bboxRef   = useRef(null); // { minLng, maxLng, minLat, maxLat }
  const pulseRef  = useRef(null);

  // ── Three.js globe (always running) ────────────────────
  useEffect(() => {
    if (!threeRef.current) return;
    return buildThreeScene(threeRef.current);
  }, []);

  // ── MapLibre: init once when metadata arrives ───────────
  useEffect(() => {
    if (!imageMetadata || !mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapRef.current,
      style: MAP_STYLE,
      center: [0, 20],
      zoom: 2,
      minZoom: 0.5,
      maxZoom: 22,
      attributionControl: false,
      pitchWithRotate: false,
    });
    mapInst.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');

    map.on('load', () => {
      mapReady.current = true;

      // Outline only — no fill inside the box
      map.addSource(SRC, { type: 'geojson', data: EMPTY });
      map.addLayer({ id: LINE, type: 'line', source: SRC,
        paint: { 'line-color': '#00ff88', 'line-width': 2.5, 'line-opacity': 0.95 } });
      map.addLayer({ id: GLOW, type: 'line', source: SRC,
        paint: { 'line-color': '#00ff88', 'line-width': 14, 'line-opacity': 0, 'line-blur': 8 } });

      const bounds = metadataToBounds(imageMetadata);
      if (bounds) {
        const [[minLat, minLng], [maxLat, maxLng]] = bounds;
        bboxRef.current = { minLng, maxLng, minLat, maxLat };
        map.getSource(SRC)?.setData({
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [[[minLng,minLat],[maxLng,minLat],[maxLng,maxLat],[minLng,maxLat],[minLng,minLat]]],
            },
            properties: {},
          }],
        });
      }
    });

    return () => {
      clearInterval(pulseRef.current);
      mapReady.current = false;
      bboxRef.current  = null;
      map.remove();
      mapInst.current = null;
    };
  }, [imageMetadata]);

  // ── Fly to site whenever viewMode switches to 'map' ─────
  useEffect(() => {
    if (viewMode !== 'map') return;
    const map  = mapInst.current;
    const bbox = bboxRef.current;
    if (!map || !bbox) return;
    const fly = () => map.flyTo({
      center:    [(bbox.minLng + bbox.maxLng) / 2, (bbox.minLat + bbox.maxLat) / 2],
      zoom:      13,
      duration:  2400,
      essential: true,
      curve:     1.4,
    });
    if (mapReady.current) fly();
    else map.once('load', fly);
  }, [viewMode]);

  // ── BBox scan animation ─────────────────────────────────
  useEffect(() => {
    const map = mapInst.current;
    clearInterval(pulseRef.current);
    if (!map || !mapReady.current) return;

    const isScanning = ['running', 'started', 'queued'].includes(jobStatus);
    const isDone     = jobStatus === 'completed';
    const color      = isDone ? '#00d4ff' : '#00ff88';

    try {
      map.setPaintProperty(LINE, 'line-color', color);
      map.setPaintProperty(GLOW, 'line-color', color);
    } catch (_) {}

    if (isScanning) {
      let t = 0;
      pulseRef.current = setInterval(() => {
        t += 0.07;
        try { map.setPaintProperty(GLOW, 'line-opacity', (0.5 + 0.5 * Math.sin(t)) * 0.45); } catch (_) {}
      }, 50);
    } else {
      try { map.setPaintProperty(GLOW, 'line-opacity', isDone ? 0.25 : 0); } catch (_) {}
    }
  }, [jobStatus]);

  const showGlobe = viewMode === 'globe' || !imageMetadata;
  const showMap   = viewMode === 'map'   && !!imageMetadata;

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 0, background: '#010203' }}>
      <div
        ref={threeRef}
        style={{
          position:  'absolute', inset: 0,
          opacity:   showGlobe ? 1 : 0,
          transition:'opacity 1.1s ease',
          pointerEvents: showGlobe ? 'auto' : 'none',
        }}
      />
      <div
        ref={mapRef}
        style={{
          position:  'absolute', inset: 0,
          opacity:   showMap ? 1 : 0,
          transition:'opacity 1.1s ease',
          pointerEvents: showMap ? 'auto' : 'none',
        }}
      />
    </div>
  );
}
