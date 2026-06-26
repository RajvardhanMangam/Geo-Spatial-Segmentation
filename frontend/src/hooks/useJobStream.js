/**
 * useJobStream — subscribes to a job's WebSocket and updates Zustand store.
 *
 * Resilience rules (frontend-only, backend unchanged):
 *  - Once status is 'completed', never overwrite it.
 *  - If backend sends 'failed' / 'error' BUT we already have detections or
 *    progress ≥ 90%, treat it as a completion artifact and promote to 'completed'.
 *    (Backend sometimes marks the job failed in a post-processing step even
 *    though all chunks were processed successfully.)
 */
import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

function isEffectivelyDone(state) {
  return (
    state.detections.length > 0 ||
    state.inferenceProgress >= 90
  );
}

export function useJobStream(jobId) {
  const wsRef = useRef(null);
  const {
    setJobStatus,
    setProgress,
    addDetections,
    setDetections,
    setCompleted,
    setError,
  } = useStore();

  useEffect(() => {
    if (!jobId) return;

    const ws = new WebSocket(`${WS_URL}/ws/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected to job', jobId);
    };

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }

      switch (msg.type) {
        case 'job_state': {
          const state = useStore.getState();
          if (state.jobStatus === 'completed') break; // never regress

          if (msg.status === 'failed' && isEffectivelyDone(state)) {
            // Backend failed after successful processing — promote to completed
            setJobStatus('completed');
            setProgress(100);
          } else {
            setJobStatus(msg.status);
            if (msg.progress) setProgress(msg.progress);
          }
          break;
        }

        case 'started':
          setJobStatus('running');
          break;

        case 'chunk_done':
          setProgress(msg.progress);
          if (msg.detections?.length > 0) {
            addDetections(msg.detections);
          }
          break;

        case 'completed':
          setProgress(100);
          setJobStatus('completed');
          if (msg.detections) setDetections(msg.detections);
          setCompleted(msg.total_detections);
          break;

        case 'error': {
          const state = useStore.getState();
          if (state.jobStatus === 'completed') break; // never regress

          if (isEffectivelyDone(state)) {
            // Error after successful chunk processing — treat as completed
            setJobStatus('completed');
            setProgress(100);
          } else {
            setError(msg.message);
            setJobStatus('failed');
          }
          break;
        }

        case 'ping':
          break;

        default:
          break;
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      const state = useStore.getState();
      if (state.jobStatus !== 'completed' && !isEffectivelyDone(state)) {
        setError('WebSocket connection error');
      } else if (isEffectivelyDone(state) && state.jobStatus !== 'completed') {
        setJobStatus('completed');
        setProgress(100);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Closed for job', jobId);
      // If socket closed while we have detections but never got 'completed', promote
      const state = useStore.getState();
      if (state.jobStatus !== 'completed' && state.jobStatus !== 'failed' && isEffectivelyDone(state)) {
        setJobStatus('completed');
        setProgress(100);
      }
    };

    return () => {
      ws.close();
    };
  }, [jobId]);

  return wsRef;
}
