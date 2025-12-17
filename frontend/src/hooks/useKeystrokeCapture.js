/**
 * React hook for capturing keystroke dynamics from Monaco Editor
 * Captures: timestamp, key, dwell time, flight time
 */

import { useEffect, useRef } from 'react';

export const useKeystrokeCapture = (editor, userId, sessionId, onDataReady) => {
  const keystrokeBuffer = useRef([]);
  const lastKeyUp = useRef(null);
  const keyDownTimes = useRef(new Map());

  useEffect(() => {
    if (!editor) return;

    // Capture keydown events (start of key press)
    const keyDownDisposable = editor.onKeyDown((e) => {
      const timestamp = Date.now();
      const key = e.browserEvent.key;

      // Store keydown time for dwell calculation
      keyDownTimes.current.set(key, timestamp);
    });

    // Capture keyup events (end of key press)
    const keyUpDisposable = editor.onKeyUp((e) => {
      const timestamp = Date.now();
      const key = e.browserEvent.key;

      // Calculate dwell time (how long key was held)
      const keyDownTime = keyDownTimes.current.get(key);
      const dwellTime = keyDownTime ? timestamp - keyDownTime : 0;

      // Calculate flight time (time since last key release)
      const flightTime = lastKeyUp.current ? timestamp - lastKeyUp.current : 0;

      // Create keystroke event
      const keystrokeEvent = {
        userId,
        sessionId,
        timestamp,
        key,
        dwellTime,
        flightTime,
        keyCode: e.browserEvent.keyCode,
      };

      // Add to buffer
      keystrokeBuffer.current.push(keystrokeEvent);

      // Update last keyup time
      lastKeyUp.current = timestamp;

      // Clean up
      keyDownTimes.current.delete(key);

      // Send batch to backend every 50 keystrokes or 10 seconds
      if (keystrokeBuffer.current.length >= 50) {
        sendBatch();
      }
    });

    // Send batch periodically
    const intervalId = setInterval(() => {
      if (keystrokeBuffer.current.length > 0) {
        sendBatch();
      }
    }, 10000); // Every 10 seconds

    const sendBatch = () => {
      if (keystrokeBuffer.current.length === 0) return;

      const batch = [...keystrokeBuffer.current];
      keystrokeBuffer.current = [];

      // Call callback with batched data
      if (onDataReady) {
        onDataReady(batch);
      }
    };

    // Cleanup
    return () => {
      keyDownDisposable.dispose();
      keyUpDisposable.dispose();
      clearInterval(intervalId);

      // Send any remaining data
      if (keystrokeBuffer.current.length > 0) {
        sendBatch();
      }
    };
  }, [editor, userId, sessionId, onDataReady]);

  return { keystrokeBuffer };
};
