import { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { FaKeyboard, FaUser, FaTrophy, FaExclamationTriangle } from 'react-icons/fa';
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture';
import authService from '../services/authService';
import './RecognizeUser.css';

const RecognizeUser = () => {
  const [editor, setEditor] = useState(null);
  const [keystrokeData, setKeystrokeData] = useState([]);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [noUsersEnrolled, setNoUsersEnrolled] = useState(false);

  const sessionId = `recognize_${Date.now()}`;
  const userId = 'recognition_user'; // Temporary user ID for capturing keystrokes

  // Handle keystroke data
  const handleKeystrokeData = (batch) => {
    setKeystrokeData((prev) => [...prev, ...batch]);
  };

  // Attach keystroke capture to editor
  useKeystrokeCapture(editor, userId, sessionId, handleKeystrokeData);

  // Check if users are enrolled on mount
  useEffect(() => {
    checkEnrolledUsers();
  }, []);

  const checkEnrolledUsers = async () => {
    try {
      const response = await authService.getEnrolledUsers();
      if (response.count === 0) {
        setNoUsersEnrolled(true);
      } else {
        setNoUsersEnrolled(false);
      }
    } catch (error) {
      console.error('Error checking enrolled users:', error);
    }
  };

  const handleEditorMount = (editorInstance) => {
    setEditor(editorInstance);
  };

  const handleRecognize = async () => {
    if (keystrokeData.length < 100) {
      setError('Please type at least 100 keystrokes for accurate recognition.');
      return;
    }

    setIsRecognizing(true);
    setError(null);
    setResults(null);

    try {
      const result = await authService.identifyUser(keystrokeData, 3);
      setResults(result);

      // Show warning for low confidence
      if (result.confidence_level === 'LOW') {
        setError('Low confidence match. Results may not be accurate.');
      }
    } catch (err) {
      console.error('Recognition error:', err);
      if (err.message && err.message.includes('No users enrolled')) {
        setNoUsersEnrolled(true);
        setError('No users enrolled yet. Please train at least one user first.');
      } else {
        setError('Recognition failed. Please ensure you have typed enough and try again.');
      }
    } finally {
      setIsRecognizing(false);
    }
  };

  const handleClear = () => {
    setKeystrokeData([]);
    setResults(null);
    setError(null);
    if (editor) {
      editor.setValue('');
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'high';
    if (confidence >= 60) return 'medium';
    return 'low';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 80) return 'HIGH';
    if (confidence >= 60) return 'MEDIUM';
    return 'LOW';
  };

  const getRankMedal = (rank) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return '';
  };

  return (
    <div className="recognize-container">
      <div className="recognize-header">
        <FaKeyboard className="header-icon" />
        <div>
          <h1>User Recognition</h1>
          <p>Type anything to identify yourself based on your typing pattern</p>
        </div>
      </div>

      {noUsersEnrolled ? (
        <div className="no-users-warning">
          <FaExclamationTriangle className="warning-icon" />
          <h2>No Users Enrolled</h2>
          <p>Please enroll at least one user before using recognition</p>
          <button
            className="button-primary"
            onClick={() => window.location.href = '/train'}
          >
            Go to Training Page
          </button>
        </div>
      ) : (
        <div className="recognize-content">
          <div className="typing-section">
            <div className="editor-header">
              <span className="keystroke-counter">
                <FaKeyboard />
                Captured: {keystrokeData.length} keystrokes
                {keystrokeData.length < 100 && (
                  <span className="counter-requirement"> (min 100 required)</span>
                )}
                {keystrokeData.length >= 100 && (
                  <span className="counter-ready"> ✓ Ready</span>
                )}
              </span>
            </div>

            <Editor
              height="400px"
              defaultLanguage="python"
              defaultValue="# Type anything here...\n# Your typing pattern will be analyzed\n\n"
              onMount={handleEditorMount}
              theme="vs-dark"
              options={{
                fontSize: 14,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                wordWrap: 'on',
              }}
            />

            <div className="action-buttons">
              <button
                className="button-primary"
                onClick={handleRecognize}
                disabled={keystrokeData.length < 100 || isRecognizing}
              >
                {isRecognizing ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <FaTrophy />
                    Recognize User
                  </>
                )}
              </button>
              <button
                className="button-secondary"
                onClick={handleClear}
                disabled={isRecognizing}
              >
                Clear & Try Again
              </button>
            </div>

            {error && (
              <div className="error-message">
                <FaExclamationTriangle />
                {error}
              </div>
            )}
          </div>

          {results && results.matches && results.matches.length > 0 && (
            <div className="results-section">
              <h2>Recognition Results</h2>
              <p className="results-info">
                Compared against {results.total_enrolled_users} enrolled user(s)
              </p>

              <div className="matches-container">
                {results.matches.map((match) => (
                  <div
                    key={match.userId}
                    className={`match-card ${getConfidenceColor(match.confidence)} ${match.rank === 1 ? 'best-match' : ''}`}
                  >
                    <div className="match-header">
                      <span className="match-rank">
                        {getRankMedal(match.rank)} Rank #{match.rank}
                        {match.rank === 1 && <span className="best-label"> - Best Match</span>}
                      </span>
                      <span className={`confidence-badge ${getConfidenceColor(match.confidence)}`}>
                        {getConfidenceLabel(match.confidence)} CONFIDENCE
                      </span>
                    </div>

                    <div className="match-user">
                      <FaUser className="user-icon" />
                      <span className="user-id">{match.userId}</span>
                    </div>

                    <div className="match-details">
                      <div className="confidence-row">
                        <span>Confidence:</span>
                        <span className="confidence-value">{match.confidence.toFixed(1)}%</span>
                      </div>
                      <div className="confidence-bar-container">
                        <div
                          className={`confidence-bar ${getConfidenceColor(match.confidence)}`}
                          style={{ width: `${match.confidence}%` }}
                        />
                      </div>
                      <div className="similarity-row">
                        <span>Similarity Score:</span>
                        <span>{match.similarity.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {results.confidence_level === 'LOW' && (
                <div className="low-confidence-warning">
                  <FaExclamationTriangle />
                  <strong>Low Confidence Warning:</strong> The best match has low confidence.
                  This could mean you're not enrolled yet, or you need to provide more typing samples during enrollment.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RecognizeUser;
