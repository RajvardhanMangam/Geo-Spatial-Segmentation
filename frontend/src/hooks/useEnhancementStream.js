/**
 * useEnhancementStream — subscribes to the road enhancement WebSocket
 * and feeds step progress + final enhanced roads into the Zustand store.
 *
 * Only connects when `active` is true (i.e. enhancement has been started).
 */
import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export function useEnhancementStream(jobId, active) {
  const wsRef = useRef(null);
  const {
    setEnhancementStatus,
    addEnhancementStep,
    setCurrentEnhancementStep,
    setEnhancedRoads,
  } = useStore();

  useEffect(() => {
    if (!jobId || !active) return;

    const ws = new WebSocket(`${WS_URL}/ws/${jobId}/enhance`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[Enhancement WS] Connected for job', jobId);
    };

    ws.onmessage = (evt) => {
      let msg;
      try { msg = JSON.parse(evt.data); } catch { return; }

      switch (msg.type) {
        case 'enhance_started':
          setEnhancementStatus('running');
          break;

        case 'enhance_step':
          setCurrentEnhancementStep(msg.step);
          addEnhancementStep(msg.step);
          break;

        case 'enhance_complete':
          setEnhancedRoads(msg.roads || []);
          setEnhancementStatus('completed');
          setCurrentEnhancementStep('');
          break;

        case 'enhance_error':
          setEnhancementStatus('failed');
          setCurrentEnhancementStep('');
          console.error('[Enhancement] Error:', msg.message);
          break;

        case 'ping':
          break;

        default:
          break;
      }
    };

    ws.onerror = (err) => {
      console.error('[Enhancement WS] Error:', err);
      setEnhancementStatus('failed');
      setCurrentEnhancementStep('');
    };

    ws.onclose = () => {
      console.log('[Enhancement WS] Closed for job', jobId);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [jobId, active]);

  return wsRef;
}
