// Shared mutable reference populated by CesiumGlobe; consumed by Header.
// Using a plain module object avoids prop-drilling and context overhead.
export const globeAPI = {
  viewer:        null,
  bounds:        null,         // { minLng, minLat, maxLng, maxLat } of current image
  imageMetadata: null,         // raw metadata from store — used as fallback for bound computation
  rotRef:        { current: true },
  orbitRef:      { current: false },
  flyToGlobe:    () => {},
  flyToArea:     () => {},
  fitBounds:     () => {},
  resetCamera:   () => {},
};
