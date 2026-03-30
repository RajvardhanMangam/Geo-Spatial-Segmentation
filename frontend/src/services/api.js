/**
 * API service — wraps all backend calls.
 */
import axios from 'axios';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const CHUNK_BYTES = 8 * 1024 * 1024; // 8 MB per chunk (matches backend)

const http = axios.create({ baseURL: API, timeout: 0 });

/**
 * Upload a large GeoTIFF in 8MB chunks.
 * @param {File} file
 * @param {function} onProgress  (0–100)
 * @returns {Promise<{uploadId, metadata}>}
 */
export async function uploadTif(file, onProgress) {
  const totalChunks = Math.ceil(file.size / CHUNK_BYTES);

  // 1. Init session
  const initForm = new FormData();
  initForm.append('filename', file.name);
  initForm.append('total_size', file.size);
  initForm.append('total_chunks', totalChunks);
  const { data: initData } = await http.post('/api/v1/upload/init', initForm);
  const uploadId = initData.upload_id;

  // 2. Send chunks
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_BYTES;
    const end = Math.min(start + CHUNK_BYTES, file.size);
    const blob = file.slice(start, end);

    const chunkForm = new FormData();
    chunkForm.append('upload_id', uploadId);
    chunkForm.append('chunk_index', i);
    chunkForm.append('chunk', blob, `chunk_${i}`);

    await http.post('/api/v1/upload/chunk', chunkForm);
    onProgress(Math.round(((i + 1) / totalChunks) * 90));
  }

  // 3. Complete
  const completeForm = new FormData();
  completeForm.append('upload_id', uploadId);
  const { data: completeData } = await http.post('/api/v1/upload/complete', completeForm);
  onProgress(100);

  return { uploadId, metadata: completeData.metadata };
}

/**
 * Start inference job.
 */
export async function startInference(uploadId, confidenceThreshold = 0.10) {
  const { data } = await http.post('/api/v1/inference/start', {
    upload_id: uploadId,
    confidence_threshold: confidenceThreshold,
  });
  return data; // { job_id, total_chunks }
}

/**
 * Get job status.
 */
export async function getJob(jobId) {
  const { data } = await http.get(`/api/v1/jobs/${jobId}`);
  return data;
}

/**
 * Download GeoJSON for a completed job.
 */
export function getGeoJsonUrl(jobId) {
  return `${API}/api/v1/jobs/${jobId}/geojson`;
}
