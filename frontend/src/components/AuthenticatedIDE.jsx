/**
 * Main IDE Component with Keystroke Authentication
 * Integrates Monaco Editor with real-time authentication monitoring
 */

import React, { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture';
import authService from '../services/authService';
import AuthStatusPanel from './AuthStatusPanel';
import AlertNotification from './AlertNotification';
import './AuthenticatedIDE.css';

const AuthenticatedIDE = ({ userId, sessionId, isExamMode = false }) => {
  const [editor, setEditor] = useState(null);
  const [code, setCode] = useState('// Start coding...\n');
  const [authStatus, setAuthStatus] = useState({
    authenticated: null,
    riskScore: 0,
    alertLevel: 'LOW',
    eventsCaptures: 0,
  });
  const [alerts, setAlerts] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const wsRef = useRef(null);

  // Handle keystroke data capture
  const handleKeystrokeData = async (batch) => {
    try {
      const result = await authService.captureKeystrokes(batch);
      setAuthStatus((prev) => ({
        ...prev,
        eventsCaptured: result.total_buffered,
      }));
    } catch (error) {
      console.error('Failed to capture keystrokes:', error);
    }
  };

  // Use keystroke capture hook
  useKeystrokeCapture(editor, userId, sessionId, handleKeystrokeData);

  // Setup WebSocket for real-time monitoring
  useEffect(() => {
    if (!isExamMode || !userId || !sessionId) return;

    wsRef.current = authService.createWebSocket(
      userId,
      sessionId,
      handleWebSocketMessage,
      handleWebSocketError
    );

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [userId, sessionId, isExamMode]);

  const handleWebSocketMessage = (data) => {
    if (data.type === 'status_update') {
      setAuthStatus({
        authenticated: data.risk_score < 0.7,
        riskScore: data.risk_score,
        alertLevel: data.risk_score > 0.7 ? 'HIGH' : data.risk_score > 0.5 ? 'MEDIUM' : 'LOW',
        eventsCaptured: data.events_captured,
      });
    } else if (data.type === 'alert') {
      addAlert({
        level: data.level,
        message: data.message,
        riskScore: data.risk_score,
      });
    }
  };

  const handleWebSocketError = (error) => {
    console.error('WebSocket error:', error);
    addAlert({
      level: 'ERROR',
      message: 'Connection to monitoring system lost',
    });
  };

  // Periodic monitoring check
  useEffect(() => {
    if (!isExamMode) return;

    const monitorInterval = setInterval(async () => {
      try {
        const result = await authService.monitorSession(userId, sessionId);

        if (result.success) {
          setAuthStatus({
            authenticated: result.authenticated,
            riskScore: result.average_risk_score,
            alertLevel: result.alert_level,
            eventsCaptured: result.samples_analyzed,
          });

          // Trigger alert if high risk
          if (result.alert_level === 'HIGH' && !alerts.find(a => a.level === 'HIGH')) {
            addAlert({
              level: 'HIGH',
              message: 'Unusual typing pattern detected. Please verify your identity.',
              riskScore: result.average_risk_score,
            });
          }
        }
      } catch (error) {
        console.error('Monitoring check failed:', error);
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(monitorInterval);
  }, [userId, sessionId, isExamMode]);

  const addAlert = (alert) => {
    const newAlert = {
      id: Date.now(),
      timestamp: new Date(),
      ...alert,
    };
    setAlerts((prev) => [newAlert, ...prev].slice(0, 10)); // Keep last 10 alerts
  };

  const handleEditorMount = (editor) => {
    setEditor(editor);
    editor.focus();
  };

  const handleEditorChange = (value) => {
    setCode(value);
  };

  // Manual verification trigger
  const handleManualVerification = async () => {
    setIsMonitoring(true);
    try {
      const result = await authService.monitorSession(userId, sessionId);

      if (result.success) {
        setAuthStatus({
          authenticated: result.authenticated,
          riskScore: result.average_risk_score,
          alertLevel: result.alert_level,
          eventsCaptured: result.samples_analyzed,
        });

        addAlert({
          level: 'INFO',
          message: `Verification complete. Risk level: ${result.alert_level}`,
          riskScore: result.average_risk_score,
        });
      }
    } catch (error) {
      addAlert({
        level: 'ERROR',
        message: 'Verification failed. Please try again.',
      });
    } finally {
      setIsMonitoring(false);
    }
  };

  return (
    <div className="authenticated-ide">
      {/* Header */}
      <div className="ide-header">
        <div className="header-left">
          <h2>Programming IDE</h2>
          {isExamMode && <span className="exam-badge">EXAM MODE</span>}
        </div>
        <div className="header-right">
          <span className="user-info">User: {userId}</span>
          <span className="session-info">Session: {sessionId}</span>
        </div>
      </div>

      <div className="ide-container">
        {/* Main Editor */}
        <div className="editor-section">
          <div className="editor-toolbar">
            <span>main.py</span>
            <button
              onClick={handleManualVerification}
              disabled={isMonitoring}
              className="verify-button"
            >
              {isMonitoring ? 'Verifying...' : 'Manual Verify'}
            </button>
          </div>

          <Editor
            height="calc(100vh - 180px)"
            defaultLanguage="python"
            value={code}
            onChange={handleEditorChange}
            onMount={handleEditorMount}
            theme="vs-dark"
            options={{
              fontSize: 14,
              minimap: { enabled: true },
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 4,
            }}
          />
        </div>

        {/* Authentication Status Panel */}
        <AuthStatusPanel
          authStatus={authStatus}
          isExamMode={isExamMode}
          alerts={alerts}
          onClearAlerts={() => setAlerts([])}
        />
      </div>

      {/* Alert Notifications */}
      <div className="alert-container">
        {alerts.slice(0, 3).map((alert) => (
          <AlertNotification
            key={alert.id}
            alert={alert}
            onClose={() => setAlerts((prev) => prev.filter((a) => a.id !== alert.id))}
          />
        ))}
      </div>
    </div>
  );
};

export default AuthenticatedIDE;
