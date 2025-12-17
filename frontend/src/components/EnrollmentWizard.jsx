/**
 * Enrollment Wizard Component
 * Guides users through the keystroke authentication enrollment process
 */

import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture';
import authService from '../services/authService';
import { FaCheckCircle, FaKeyboard, FaUserCheck } from 'react-icons/fa';
import './EnrollmentWizard.css';

const ENROLLMENT_EXERCISES = [
  {
    id: 1,
    title: 'Hello World',
    description: 'Type a simple Python program',
    template: '# Write a Hello World program\n',
    minKeystrokes: 50,
  },
  {
    id: 2,
    title: 'Function Definition',
    description: 'Create a function that adds two numbers',
    template: '# Define a function called add_numbers\n',
    minKeystrokes: 50,
  },
  {
    id: 3,
    title: 'Loop Practice',
    description: 'Write a for loop that prints numbers 1 to 10',
    template: '# Write a for loop\n',
    minKeystrokes: 50,
  },
  {
    id: 4,
    title: 'Free Typing',
    description: 'Type any code you like for at least 30 seconds',
    template: '# Write any Python code\n',
    minKeystrokes: 100,
  },
];

const EnrollmentWizard = ({ userId, onEnrollmentComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [editor, setEditor] = useState(null);
  const [code, setCode] = useState(ENROLLMENT_EXERCISES[0].template);
  const [keystrokeData, setKeystrokeData] = useState([]);
  const [exerciseProgress, setExerciseProgress] = useState(
    ENROLLMENT_EXERCISES.map(() => ({ completed: false, keystrokes: 0 }))
  );
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentStatus, setEnrollmentStatus] = useState(null);

  const sessionId = `enrollment_${Date.now()}`;

  // Handle keystroke capture
  const handleKeystrokeData = (batch) => {
    setKeystrokeData((prev) => [...prev, ...batch]);

    // Update progress for current exercise
    setExerciseProgress((prev) => {
      const newProgress = [...prev];
      newProgress[currentStep] = {
        ...newProgress[currentStep],
        keystrokes: prev[currentStep].keystrokes + batch.length,
      };
      return newProgress;
    });
  };

  // Use keystroke capture hook
  useKeystrokeCapture(editor, userId, sessionId, handleKeystrokeData);

  const handleEditorMount = (editor) => {
    setEditor(editor);
    editor.focus();
  };

  const handleEditorChange = (value) => {
    setCode(value);
  };

  const currentExercise = ENROLLMENT_EXERCISES[currentStep];
  const currentProgress = exerciseProgress[currentStep];
  const canProceed = currentProgress.keystrokes >= currentExercise.minKeystrokes;

  const handleNext = () => {
    if (currentStep < ENROLLMENT_EXERCISES.length - 1) {
      // Mark current as completed
      setExerciseProgress((prev) => {
        const newProgress = [...prev];
        newProgress[currentStep].completed = true;
        return newProgress;
      });

      // Move to next exercise
      setCurrentStep(currentStep + 1);
      setCode(ENROLLMENT_EXERCISES[currentStep + 1].template);
    } else {
      // All exercises complete, proceed to enrollment
      handleEnrollment();
    }
  };

  const handleEnrollment = async () => {
    setIsEnrolling(true);

    try {
      // Ensure we have enough data
      if (keystrokeData.length < 200) {
        setEnrollmentStatus({
          success: false,
          message: 'Not enough typing data collected. Please complete more exercises.',
        });
        setIsEnrolling(false);
        return;
      }

      // Send enrollment request
      const result = await authService.enrollUser(userId, keystrokeData);

      if (result.success) {
        setEnrollmentStatus({
          success: true,
          message: 'Enrollment successful! You are now authenticated.',
          details: result,
        });

        // Call completion callback after delay
        setTimeout(() => {
          if (onEnrollmentComplete) {
            onEnrollmentComplete(result);
          }
        }, 2000);
      } else {
        setEnrollmentStatus({
          success: false,
          message: 'Enrollment failed. Please try again.',
        });
      }
    } catch (error) {
      setEnrollmentStatus({
        success: false,
        message: `Error: ${error.message}`,
      });
    } finally {
      setIsEnrolling(false);
    }
  };

  const progressPercentage = (currentProgress.keystrokes / currentExercise.minKeystrokes) * 100;

  if (enrollmentStatus) {
    return (
      <div className="enrollment-wizard">
        <div className="enrollment-complete">
          {enrollmentStatus.success ? (
            <>
              <FaUserCheck className="success-icon" />
              <h2>Enrollment Complete!</h2>
              <p>{enrollmentStatus.message}</p>
              <div className="enrollment-details">
                <p>Total keystrokes collected: {keystrokeData.length}</p>
                <p>User ID: {userId}</p>
              </div>
            </>
          ) : (
            <>
              <div className="error-icon">×</div>
              <h2>Enrollment Failed</h2>
              <p>{enrollmentStatus.message}</p>
              <button onClick={() => setEnrollmentStatus(null)} className="retry-button">
                Try Again
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="enrollment-wizard">
      {/* Header */}
      <div className="wizard-header">
        <h2>
          <FaKeyboard /> Keystroke Authentication Enrollment
        </h2>
        <p>Complete typing exercises to create your authentication profile</p>
      </div>

      {/* Progress Steps */}
      <div className="wizard-steps">
        {ENROLLMENT_EXERCISES.map((exercise, index) => (
          <div
            key={exercise.id}
            className={`step ${index === currentStep ? 'active' : ''} ${
              exerciseProgress[index].completed ? 'completed' : ''
            }`}
          >
            <div className="step-number">
              {exerciseProgress[index].completed ? <FaCheckCircle /> : index + 1}
            </div>
            <div className="step-title">{exercise.title}</div>
          </div>
        ))}
      </div>

      {/* Current Exercise */}
      <div className="wizard-content">
        <div className="exercise-info">
          <h3>{currentExercise.title}</h3>
          <p>{currentExercise.description}</p>

          {/* Progress Bar */}
          <div className="progress-container">
            <div className="progress-label">
              <span>Progress: {currentProgress.keystrokes} / {currentExercise.minKeystrokes} keystrokes</span>
              <span>{Math.min(100, progressPercentage).toFixed(0)}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${Math.min(100, progressPercentage)}%`,
                  backgroundColor: canProceed ? '#27ae60' : '#3498db',
                }}
              />
            </div>
          </div>
        </div>

        {/* Code Editor */}
        <div className="exercise-editor">
          <Editor
            height="400px"
            defaultLanguage="python"
            value={code}
            onChange={handleEditorChange}
            onMount={handleEditorMount}
            theme="vs-dark"
            options={{
              fontSize: 14,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </div>

        {/* Controls */}
        <div className="wizard-controls">
          {currentStep > 0 && (
            <button
              onClick={() => {
                setCurrentStep(currentStep - 1);
                setCode(ENROLLMENT_EXERCISES[currentStep - 1].template);
              }}
              className="button-secondary"
            >
              Previous
            </button>
          )}

          <div className="control-info">
            {canProceed ? (
              <span className="success-text">
                <FaCheckCircle /> Ready to proceed!
              </span>
            ) : (
              <span className="info-text">
                Type at least {currentExercise.minKeystrokes - currentProgress.keystrokes} more characters
              </span>
            )}
          </div>

          <button
            onClick={handleNext}
            disabled={!canProceed || isEnrolling}
            className="button-primary"
          >
            {isEnrolling
              ? 'Enrolling...'
              : currentStep === ENROLLMENT_EXERCISES.length - 1
              ? 'Complete Enrollment'
              : 'Next Exercise'}
          </button>
        </div>
      </div>

      {/* Info Footer */}
      <div className="wizard-footer">
        <p>
          <strong>Why enrollment?</strong> We analyze your unique typing patterns (timing and
          rhythm) to create a behavioral biometric profile. This ensures only you can access your
          exams.
        </p>
        <p className="privacy-note">
          🔒 Only typing timing data is stored, not the actual content you type.
        </p>
      </div>
    </div>
  );
};

export default EnrollmentWizard;
