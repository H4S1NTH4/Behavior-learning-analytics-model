/**
 * Enrollment Wizard Component
 * Guides users through the keystroke authentication enrollment process
 */

import { useState } from 'react';
import TypingTest from './TypingTest';
import authService from '../services/authService';
import { FaCheckCircle, FaKeyboard, FaUserCheck } from 'react-icons/fa';
import './EnrollmentWizard.css';

const ENROLLMENT_EXERCISES = [
  {
    id: 1,
    title: 'Hello World',
    description: 'Type the following Python program exactly as shown',
    template: `def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}! Welcome to Python."

# Main execution
if __name__ == "__main__":
    user_name = input("Enter your name: ")
    message = greet(user_name)
    print(message)`,
    minKeystrokes: 50,
  },
  {
    id: 2,
    title: 'Function Definition',
    description: 'Type this function that performs mathematical operations',
    template: `def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0

    total = sum(numbers)
    count = len(numbers)
    average = total / count

    return round(average, 2)

# Example usage
scores = [85, 92, 78, 90, 88]
result = calculate_average(scores)
print(f"Average score: {result}")`,
    minKeystrokes: 50,
  },
  {
    id: 3,
    title: 'Loop Practice',
    description: 'Type this code that demonstrates loops and conditionals',
    template: `def find_even_numbers(start, end):
    """Find all even numbers in a given range."""
    even_nums = []

    for num in range(start, end + 1):
        if num % 2 == 0:
            even_nums.append(num)

    return even_nums

# Test the function
numbers = find_even_numbers(1, 20)
print(f"Even numbers: {numbers}")
print(f"Count: {len(numbers)}")`,
    minKeystrokes: 50,
  },
  {
    id: 4,
    title: 'Class Definition',
    description: 'Type this class implementation with methods',
    template: `class Student:
    """Represents a student with name and grades."""

    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = []

    def add_grade(self, grade):
        """Add a grade to the student's record."""
        if 0 <= grade <= 100:
            self.grades.append(grade)
            return True
        return False

    def get_average(self):
        """Calculate the student's average grade."""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)

# Create student instance
student = Student("Alice", "S12345")
student.add_grade(95)
student.add_grade(88)
print(f"{student.name}'s average: {student.get_average():.2f}")`,
    minKeystrokes: 100,
  },
];

const EnrollmentWizard = ({ userId, onEnrollmentComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [keystrokeData, setKeystrokeData] = useState([]);
  const [exerciseProgress, setExerciseProgress] = useState(
    ENROLLMENT_EXERCISES.map(() => ({ completed: false, keystrokes: 0 }))
  );
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentStatus, setEnrollmentStatus] = useState(null);

  const sessionId = `enrollment_${Date.now()}`;

  // Handle individual keystroke events
  const handleKeystroke = (keystrokeEvent) => {
    setKeystrokeData((prev) => [...prev, keystrokeEvent]);

    // Update progress for current exercise
    setExerciseProgress((prev) => {
      const newProgress = [...prev];
      newProgress[currentStep] = {
        ...newProgress[currentStep],
        keystrokes: prev[currentStep].keystrokes + 1,
      };
      return newProgress;
    });
  };

  const handleExerciseComplete = () => {
    // Exercise completed, do nothing - user will click Next
    console.log('Exercise completed');
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

        {/* Typing Test */}
        <div className="exercise-editor">
          <TypingTest
            key={currentStep}
            targetText={currentExercise.template}
            onKeystroke={handleKeystroke}
            onComplete={handleExerciseComplete}
            userId={userId}
            sessionId={sessionId}
          />
        </div>

        {/* Controls */}
        <div className="wizard-controls">
          {currentStep > 0 && (
            <button
              onClick={() => {
                setCurrentStep(currentStep - 1);
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
