/**
 * Alert Notification Component
 * Toast-style notifications for authentication events
 */

import React, { useEffect } from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaInfoCircle, FaTimesCircle } from 'react-icons/fa';
import './AlertNotification.css';

const AlertNotification = ({ alert, onClose, autoClose = true, duration = 5000 }) => {
  useEffect(() => {
    if (autoClose && alert.level !== 'HIGH') {
      const timer = setTimeout(() => {
        onClose();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [autoClose, duration, onClose, alert.level]);

  const getIcon = () => {
    switch (alert.level) {
      case 'HIGH':
        return <FaExclamationTriangle className="alert-icon danger" />;
      case 'MEDIUM':
        return <FaExclamationTriangle className="alert-icon warning" />;
      case 'LOW':
        return <FaCheckCircle className="alert-icon success" />;
      case 'INFO':
        return <FaInfoCircle className="alert-icon info" />;
      case 'ERROR':
        return <FaTimesCircle className="alert-icon error" />;
      default:
        return <FaInfoCircle className="alert-icon" />;
    }
  };

  const getAlertClass = () => {
    switch (alert.level) {
      case 'HIGH':
        return 'alert-notification danger';
      case 'MEDIUM':
        return 'alert-notification warning';
      case 'LOW':
        return 'alert-notification success';
      case 'INFO':
        return 'alert-notification info';
      case 'ERROR':
        return 'alert-notification error';
      default:
        return 'alert-notification';
    }
  };

  return (
    <div className={getAlertClass()}>
      <div className="alert-content">
        {getIcon()}
        <div className="alert-text">
          <div className="alert-title">
            {alert.level === 'HIGH' && 'Security Alert'}
            {alert.level === 'MEDIUM' && 'Warning'}
            {alert.level === 'LOW' && 'Success'}
            {alert.level === 'INFO' && 'Information'}
            {alert.level === 'ERROR' && 'Error'}
          </div>
          <div className="alert-message">{alert.message}</div>
          {alert.riskScore !== undefined && (
            <div className="alert-details">Risk Score: {(alert.riskScore * 100).toFixed(0)}%</div>
          )}
        </div>
      </div>
      <button className="alert-close" onClick={onClose}>
        ×
      </button>
    </div>
  );
};

export default AlertNotification;
