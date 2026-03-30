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

  // Detections — array of detection objects
  detections: [],

  // Active filters
  activeFilters: {
    building: true,
    road: true,
    utility: true,
    vegetation: true,
    water: true,
  },

  // Confidence threshold
  confidenceThreshold: 0.10,

  // ── Actions ──────────────────────────────────────────────────
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

  clearDetections: () => set({ detections: [] }),

  toggleFilter: (type) =>
    set((state) => ({
      activeFilters: {
        ...state.activeFilters,
        [type]: !state.activeFilters[type],
      },
    })),

  setConfidenceThreshold: (v) => set({ confidenceThreshold: v }),

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
    }),
}));
