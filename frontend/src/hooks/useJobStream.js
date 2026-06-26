/**
 * useJobStream — subscribes to a job's WebSocket and updates Zustand store.
 */
import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';
import { getDetections, getJob } from '../services/api';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

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

    let cancelled = false;
    let syncedFinal = false;
    let reconnectTimer = null;
    let poll = null;
    let terminal = false;

    const stopLiveUpdates = () => {
      terminal = true;
      if (poll) {
        window.clearInterval(poll);
        poll = null;
      }
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      wsRef.current?.close();
    };

    const syncCompletedJob = async () => {
      if (syncedFinal || cancelled) return;
      syncedFinal = true;
      try {
        const { detections } = await getDetections(jobId);
        if (cancelled) return;
        setDetections(detections || []);
        setCompleted((detections || []).length);
        setProgress(100);
        setJobStatus('completed');
        stopLiveUpdates();
      } catch (err) {
        syncedFinal = false;
        if (!cancelled) {
          console.error('[WS] Failed to fetch final detections:', err);
        }
      }
    };

    const syncCurrentDetections = async () => {
      try {
        const { detections } = await getDetections(jobId);
        if (cancelled || syncedFinal) return;
        setDetections(detections || []);
        setCompleted((detections || []).length);
      } catch (err) {
        if (!cancelled) {
          console.error('[WS] Failed to fetch current detections:', err);
        }
      }
    };

    const syncJobState = async () => {
      try {
        const job = await getJob(jobId);
        if (cancelled || terminal) return;
        setJobStatus(job.status);
        if (typeof job.progress === 'number') setProgress(job.progress);
        if (job.status === 'completed') {
          await syncCompletedJob();
        } else if (job.status === 'running' || job.status === 'queued') {
          await syncCurrentDetections();
        } else if (job.status === 'failed') {
          setError(job.error || 'Inference failed');
          stopLiveUpdates();
        }
      } catch (err) {
        if (!cancelled) {
          console.error('[WS] Failed to fetch job state:', err);
        }
      }
    };

    const connectWebSocket = () => {
      if (cancelled || syncedFinal || terminal) return;

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
            if (typeof msg.progress === 'number') setProgress(msg.progress);
            if (msg.status === 'completed') syncCompletedJob();
            if (msg.status === 'failed') stopLiveUpdates();
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

        case 'detections_snapshot':
          setDetections(msg.detections || []);
          setCompleted(msg.count || (msg.detections || []).length);
          break;

        case 'completed':
            if (msg.detections) {
              setDetections(msg.detections);
              setCompleted(msg.total_detections || msg.detections.length);
              setProgress(100);
              setJobStatus('completed');
              syncedFinal = true;
              stopLiveUpdates();
            } else {
              syncCompletedJob();
            }
            break;

          case 'error':
            setError(msg.message);
            setJobStatus('failed');
            stopLiveUpdates();
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
      };

      ws.onclose = () => {
        console.log('[WS] Closed for job', jobId);
        if (!terminal) syncJobState();
        if (!cancelled && !syncedFinal && !terminal) {
          reconnectTimer = window.setTimeout(connectWebSocket, 3000);
        }
      };
    };

    connectWebSocket();

    poll = window.setInterval(syncJobState, 5000);
    syncJobState();

    return () => {
      cancelled = true;
      stopLiveUpdates();
    };
  }, [jobId]);

  return wsRef;
}
