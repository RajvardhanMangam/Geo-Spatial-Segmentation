/**
 * Global application state with Zustand.
 */
import { create } from 'zustand';

export const useStore = create((set, get) => ({
  // Upload state
  uploadId: null,
  uploadProgress: 0,
  uploadStatus: 'idle', // idle | uploading | complete | error
  imageMetadata: null,

  // Job / inference state
  jobId: null,
  jobStatus: 'idle',   // idle | queued | running | completed | failed
  inferenceProgress: 0,
  totalDetections: 0,
  error: null,

  // Detections — array of detection objects (original merged roads)
  detections: [],

  // Road Enhancement Mode
  roadMode: 'original',          // 'original' | 'enhanced'
  enhancementStatus: 'idle',     // 'idle' | 'running' | 'completed' | 'failed'
  enhancementSteps: [],          // list of step names that have completed
  currentEnhancementStep: '',    // name of step currently executing
  enhancedRoads: [],             // enhanced road detections from user-triggered pipeline

  // Active filters
  activeFilters: {
    building: true,
    road: true,
    water: true,
  },

  // Confidence threshold
  confidenceThreshold: 0.10,

  // UI status message (derived from job phase for display)
  statusMessage: '',

  // ── Inference actions ─────────────────────────────────────────
  setUploadId: (id) => set({ uploadId: id }),
  setUploadProgress: (p) => set({ uploadProgress: p }),
  setUploadStatus: (s) => set({ uploadStatus: s }),
  setImageMetadata: (m) => set({ imageMetadata: m }),

  setJobId: (id) => set({ jobId: id }),
  setJobStatus: (s) => set({ jobStatus: s }),
  setProgress: (p) => set({ inferenceProgress: p }),
  setCompleted: (total) => set({ totalDetections: total }),
  setError: (e) => set({ error: e }),

  addDetections: (newDets) =>
    set((state) => ({ detections: [...state.detections, ...newDets] })),

  setDetections: (detections) => set({ detections }),

  clearDetections: () => set({ detections: [] }),

  toggleFilter: (type) =>
    set((state) => ({
      activeFilters: {
        ...state.activeFilters,
        [type]: !state.activeFilters[type],
      },
    })),

  setConfidenceThreshold: (v) => set({ confidenceThreshold: v }),
  setStatusMessage: (msg) => set({ statusMessage: msg }),

  // ── Road enhancement actions ──────────────────────────────────
  setRoadMode: (mode) => set({ roadMode: mode }),

  setEnhancementStatus: (s) => set({ enhancementStatus: s }),

  addEnhancementStep: (step) =>
    set((state) => ({
      enhancementSteps: state.enhancementSteps.includes(step)
        ? state.enhancementSteps
        : [...state.enhancementSteps, step],
    })),

  setCurrentEnhancementStep: (step) => set({ currentEnhancementStep: step }),

  setEnhancedRoads: (roads) => set({ enhancedRoads: roads }),

  resetEnhancement: () =>
    set({
      roadMode: 'original',
      enhancementStatus: 'idle',
      enhancementSteps: [],
      currentEnhancementStep: '',
      enhancedRoads: [],
    }),

  // ── Full reset ────────────────────────────────────────────────
  reset: () =>
    set({
      uploadId: null,
      uploadProgress: 0,
      uploadStatus: 'idle',
      imageMetadata: null,
      jobId: null,
      jobStatus: 'idle',
      inferenceProgress: 0,
      totalDetections: 0,
      error: null,
      detections: [],
      statusMessage: '',
      roadMode: 'original',
      enhancementStatus: 'idle',
      enhancementSteps: [],
      currentEnhancementStep: '',
      enhancedRoads: [],
    }),
}));
