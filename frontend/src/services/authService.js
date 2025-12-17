/**
 * API Service for Keystroke Authentication
 * Handles all backend communication
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

class AuthService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Capture keystroke events
   */
  async captureKeystrokes(events) {
    try {
      const response = await this.api.post('/api/keystroke/capture', {
        events,
      });
      return response.data;
    } catch (error) {
      console.error('Error capturing keystrokes:', error);
      throw error;
    }
  }

  /**
   * Enroll a user with keystroke data
   */
  async enrollUser(userId, keystrokeEvents) {
    try {
      const response = await this.api.post('/api/auth/enroll', {
        userId,
        keystrokeEvents,
      });
      return response.data;
    } catch (error) {
      console.error('Error enrolling user:', error);
      throw error;
    }
  }

  /**
   * Verify a user's identity
   */
  async verifyUser(userId, keystrokeEvents, threshold = 0.7) {
    try {
      const response = await this.api.post('/api/auth/verify', {
        userId,
        keystrokeEvents,
        threshold,
      });
      return response.data;
    } catch (error) {
      console.error('Error verifying user:', error);
      throw error;
    }
  }

  /**
   * Identify a user from keystroke pattern
   * Compares against all enrolled users and returns top matches
   */
  async identifyUser(keystrokeEvents, topK = 3) {
    try {
      const response = await this.api.post('/api/auth/identify', {
        keystrokeEvents,
        topK,
      });
      return response.data;
    } catch (error) {
      console.error('Error identifying user:', error);
      // Check if no users enrolled
      if (error.response?.status === 404) {
        throw new Error('No users enrolled yet. Please train at least one user first.');
      }
      throw error;
    }
  }

  /**
   * Monitor a session for continuous authentication
   */
  async monitorSession(userId, sessionId) {
    try {
      const response = await this.api.post('/api/auth/monitor', {
        userId,
        sessionId,
      });
      return response.data;
    } catch (error) {
      console.error('Error monitoring session:', error);
      throw error;
    }
  }

  /**
   * Get session status
   */
  async getSessionStatus(userId, sessionId) {
    try {
      const response = await this.api.get(
        `/api/session/status/${userId}/${sessionId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error getting session status:', error);
      throw error;
    }
  }

  /**
   * End a session
   */
  async endSession(userId, sessionId) {
    try {
      const response = await this.api.delete(
        `/api/session/${userId}/${sessionId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error ending session:', error);
      throw error;
    }
  }

  /**
   * Get list of enrolled users
   */
  async getEnrolledUsers() {
    try {
      const response = await this.api.get('/api/users/enrolled');
      return response.data;
    } catch (error) {
      console.error('Error getting enrolled users:', error);
      throw error;
    }
  }

  /**
   * Create WebSocket connection for real-time monitoring
   */
  createWebSocket(userId, sessionId, onMessage, onError) {
    const wsUrl = `ws://localhost:8001/ws/monitor/${userId}/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return ws;
  }
}

export default new AuthService();
