import { useState, useEffect } from 'react';
import { FaUser, FaCheckCircle, FaExclamationCircle, FaKeyboard, FaArrowRight } from 'react-icons/fa';
import EnrollmentWizard from './EnrollmentWizard';
import authService from '../services/authService';
import './TrainUser.css';

const TrainUser = () => {
  const [username, setUsername] = useState('');
  const [userId, setUserId] = useState('');
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentComplete, setEnrollmentComplete] = useState(false);
  const [enrolledUsers, setEnrolledUsers] = useState([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchEnrolledUsers();
  }, []);

  const fetchEnrolledUsers = async () => {
    try {
      const response = await authService.getEnrolledUsers();
      setEnrolledUsers(response.users || []);
    } catch (error) {
      console.error('Error fetching enrolled users:', error);
    }
  };

  const validateUsername = (name) => {
    const regex = /^[a-zA-Z0-9_]{3,20}$/;
    return regex.test(name);
  };

  const handleStartTraining = () => {
    setError('');

    if (!username.trim()) {
      setError('Please enter a username');
      return;
    }

    if (!validateUsername(username)) {
      setError('Username must be 3-20 characters long and contain only letters, numbers, and underscores');
      return;
    }

    // Check if username already exists
    if (enrolledUsers.includes(username)) {
      setError('This username is already enrolled. Please choose a different name.');
      return;
    }

    // Generate user ID and start enrollment
    setUserId(username);
    setIsEnrolling(true);
  };

  const handleEnrollmentComplete = async (success, message) => {
    if (success) {
      setEnrollmentComplete(true);
      // Refresh enrolled users list
      await fetchEnrolledUsers();
    } else {
      setError(message || 'Enrollment failed. Please try again.');
      setIsEnrolling(false);
    }
  };

  const handleTrainAnother = () => {
    setUsername('');
    setUserId('');
    setIsEnrolling(false);
    setEnrollmentComplete(false);
    setError('');
  };

  const handleGoToRecognition = () => {
    window.location.href = '/recognize';
  };

  if (enrollmentComplete) {
    return (
      <div className="train-user-container">
        <div className="success-screen">
          <FaCheckCircle className="success-icon" />
          <h1>Training Complete!</h1>
          <p>User <strong>{username}</strong> has been successfully enrolled.</p>
          <p className="success-info">
            The system has learned your typing pattern. You can now use the Recognition page to test identification.
          </p>
          <div className="success-actions">
            <button className="button-primary" onClick={handleGoToRecognition}>
              <FaKeyboard />
              Go to Recognition
            </button>
            <button className="button-secondary" onClick={handleTrainAnother}>
              <FaUser />
              Train Another User
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isEnrolling) {
    return (
      <div className="train-user-container training-active">
        <EnrollmentWizard
          userId={userId}
          onEnrollmentComplete={handleEnrollmentComplete}
        />
      </div>
    );
  }

  return (
    <div className="train-user-container">
      <div className="train-user-header">
        <FaUser className="header-icon" />
        <div>
          <h1>Train New User</h1>
          <p>Enter a username and complete typing exercises to enroll a new user</p>
        </div>
      </div>

      <div className="train-user-content">
        <div className="username-section">
          <div className="username-card">
            <h2>User Information</h2>

            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                type="text"
                className="username-input"
                placeholder="Enter username (e.g., student_alice)"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setError('');
                }}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleStartTraining();
                  }
                }}
                disabled={isLoading}
              />
              <p className="input-hint">
                3-20 characters, letters, numbers, and underscores only
              </p>
            </div>

            {error && (
              <div className="error-message">
                <FaExclamationCircle />
                {error}
              </div>
            )}

            <button
              className="button-primary start-button"
              onClick={handleStartTraining}
              disabled={!username || isLoading}
            >
              <FaKeyboard />
              Start Training
              <FaArrowRight />
            </button>

            <div className="training-info">
              <h3>What to Expect:</h3>
              <ul>
                <li>Complete 4 typing exercises</li>
                <li>Minimum 250 keystrokes total</li>
                <li>Takes about 3-5 minutes</li>
                <li>Type naturally - don't rush!</li>
              </ul>
            </div>
          </div>
        </div>

        {enrolledUsers.length > 0 && (
          <div className="enrolled-users-section">
            <h2>Enrolled Users ({enrolledUsers.length})</h2>
            <div className="enrolled-users-list">
              {enrolledUsers.map((user, index) => (
                <div key={index} className="enrolled-user-item">
                  <FaCheckCircle className="enrolled-icon" />
                  <span>{user}</span>
                </div>
              ))}
            </div>
            <p className="enrolled-hint">
              These users can be recognized on the Recognition page
            </p>
          </div>
        )}

        {enrolledUsers.length === 0 && (
          <div className="no-users-info">
            <FaUser className="no-users-icon" />
            <h3>No Users Enrolled Yet</h3>
            <p>Be the first to train the keystroke authentication system!</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainUser;
