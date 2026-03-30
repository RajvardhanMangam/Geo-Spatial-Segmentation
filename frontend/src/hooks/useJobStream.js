/**
 * useJobStream — subscribes to a job's WebSocket and updates Zustand store.
 */
import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export function useJobStream(jobId) {
  const wsRef = useRef(null);
  const {
    setJobStatus,
    setProgress,
    addDetections,
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
        case 'job_state':
          setJobStatus(msg.status);
          if (msg.progress) setProgress(msg.progress);
          break;

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
          if (msg.detections) addDetections(msg.detections);
          setCompleted(msg.total_detections);
          break;

        case 'error':
          setError(msg.message);
          setJobStatus('failed');
          break;

        case 'ping':
          // keepalive, ignore
          break;

        default:
          break;
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('[WS] Closed for job', jobId);
    };

    return () => {
      ws.close();
    };
  }, [jobId]);

  return wsRef;
}
