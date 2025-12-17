/**
 * Authentication Status Panel
 * Displays real-time authentication status and risk metrics
 */

import React from 'react';
import { FaShieldAlt, FaExclamationTriangle, FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import './AuthStatusPanel.css';

const AuthStatusPanel = ({ authStatus, isExamMode, alerts, onClearAlerts }) => {
  const { authenticated, riskScore, alertLevel, eventsCaptured } = authStatus;

  const getStatusIcon = () => {
    if (authenticated === null) return <FaShieldAlt className="icon-neutral" />;
    if (authenticated) return <FaCheckCircle className="icon-success" />;
    return <FaTimesCircle className="icon-danger" />;
  };

  const getStatusText = () => {
    if (authenticated === null) return 'Initializing...';
    if (authenticated) return 'Authenticated';
    return 'Authentication Failed';
  };

  const getAlertLevelClass = () => {
    switch (alertLevel) {
      case 'HIGH':
        return 'alert-high';
      case 'MEDIUM':
        return 'alert-medium';
      case 'LOW':
        return 'alert-low';
      default:
        return '';
    }
  };

  const getRiskColor = () => {
    if (riskScore > 0.7) return '#e74c3c';
    if (riskScore > 0.5) return '#f39c12';
    return '#27ae60';
  };

  return (
    <div className="auth-status-panel">
      <div className="panel-header">
        <h3>Authentication Status</h3>
        {isExamMode && <span className="exam-indicator">MONITORED</span>}
      </div>

      {/* Authentication Status */}
      <div className="status-card">
        <div className="status-icon">{getStatusIcon()}</div>
        <div className="status-info">
          <div className="status-label">Status</div>
          <div className={`status-value ${authenticated ? 'success' : 'danger'}`}>
            {getStatusText()}
          </div>
        </div>
      </div>

      {/* Risk Score */}
      <div className="metric-card">
        <div className="metric-label">Risk Score</div>
        <div className="risk-score-container">
          <div className="risk-score-bar">
            <div
              className="risk-score-fill"
              style={{
                width: `${riskScore * 100}%`,
                backgroundColor: getRiskColor(),
              }}
            />
          </div>
          <div className="risk-score-value" style={{ color: getRiskColor() }}>
            {(riskScore * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Alert Level */}
      <div className="metric-card">
        <div className="metric-label">Alert Level</div>
        <div className={`alert-badge ${getAlertLevelClass()}`}>
          {alertLevel === 'HIGH' && <FaExclamationTriangle />}
          {alertLevel}
        </div>
      </div>

      {/* Events Captured */}
      <div className="metric-card">
        <div className="metric-label">Events Captured</div>
        <div className="metric-value">{eventsCaptured || 0}</div>
        <div className="metric-subtitle">
          {eventsCaptured < 100 ? 'Collecting baseline data...' : 'Monitoring active'}
        </div>
      </div>

      {/* Risk Indicator */}
      <div className="risk-indicator">
        <div className="risk-levels">
          <div className={`risk-level ${riskScore < 0.5 ? 'active' : ''}`}>
            <div className="risk-dot" style={{ backgroundColor: '#27ae60' }} />
            <span>Low Risk</span>
          </div>
          <div className={`risk-level ${riskScore >= 0.5 && riskScore < 0.7 ? 'active' : ''}`}>
            <div className="risk-dot" style={{ backgroundColor: '#f39c12' }} />
            <span>Medium Risk</span>
          </div>
          <div className={`risk-level ${riskScore >= 0.7 ? 'active' : ''}`}>
            <div className="risk-dot" style={{ backgroundColor: '#e74c3c' }} />
            <span>High Risk</span>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          <div className="alerts-header">
            <span>Recent Alerts ({alerts.length})</span>
            <button onClick={onClearAlerts} className="clear-button">
              Clear
            </button>
          </div>
          <div className="alerts-list">
            {alerts.slice(0, 5).map((alert) => (
              <div key={alert.id} className={`alert-item alert-${alert.level.toLowerCase()}`}>
                <div className="alert-time">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </div>
                <div className="alert-message">{alert.message}</div>
                {alert.riskScore !== undefined && (
                  <div className="alert-risk">Risk: {(alert.riskScore * 100).toFixed(0)}%</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Panel */}
      {isExamMode && (
        <div className="info-panel">
          <FaShieldAlt className="info-icon" />
          <p className="info-text">
            Your typing patterns are being monitored for authentication purposes. This ensures exam
            integrity while protecting your work.
          </p>
        </div>
      )}
    </div>
  );
};

export default AuthStatusPanel;
