/**
 * Typing Test Component
 * Displays text to be typed with character-by-character highlighting
 * Similar to Monkeytype/Keybr interface
 */

import { useState, useEffect, useRef } from 'react';
import './TypingTest.css';

const TypingTest = ({
  targetText,
  onKeystroke,
  onComplete,
  userId,
  sessionId
}) => {
  const [userInput, setUserInput] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [errors, setErrors] = useState(new Set());
  const [isFocused, setIsFocused] = useState(false);

  const inputRef = useRef(null);
  const lastKeyUp = useRef(null);
  const keyDownTimes = useRef(new Map());

  useEffect(() => {
    // Auto focus on mount
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  useEffect(() => {
    // Check if test is complete
    if (userInput.length === targetText.length && userInput === targetText) {
      if (onComplete) {
        onComplete();
      }
    }
  }, [userInput, targetText, onComplete]);

  const handleKeyDown = (e) => {
    const timestamp = Date.now();
    const key = e.key;

    // Prevent default Tab behavior to allow typing tabs
    if (key === 'Tab') {
      e.preventDefault();
    }

    // Handle Enter key for auto-indentation
    if (key === 'Enter') {
      e.preventDefault();

      // Find the current line in the target text
      const currentPos = userInput.length;
      const textUpToCursor = targetText.substring(0, currentPos);
      const lastNewlineIndex = textUpToCursor.lastIndexOf('\n');
      const currentLine = textUpToCursor.substring(lastNewlineIndex + 1);

      // Count leading spaces in current line
      const indentMatch = currentLine.match(/^(\s+)/);
      const currentIndent = indentMatch ? indentMatch[1] : '';

      // Find the next line in target text
      const nextNewlineIndex = targetText.indexOf('\n', currentPos);
      if (nextNewlineIndex !== -1) {
        const nextLineStart = nextNewlineIndex + 1;
        const nextLineText = targetText.substring(nextLineStart);
        const nextLineIndentMatch = nextLineText.match(/^(\s+)/);
        const nextIndent = nextLineIndentMatch ? nextLineIndentMatch[1] : '';

        // Insert newline with the next line's indentation
        const newInput = userInput + '\n' + nextIndent;
        setUserInput(newInput);
        setCurrentIndex(newInput.length);

        // Update errors
        const newErrors = new Set();
        for (let i = 0; i < newInput.length; i++) {
          if (newInput[i] !== targetText[i]) {
            newErrors.add(i);
          }
        }
        setErrors(newErrors);

        // Set cursor position after state update
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.setSelectionRange(newInput.length, newInput.length);
          }
        }, 0);
      } else {
        // Just add newline if no next line
        const newInput = userInput + '\n';
        setUserInput(newInput);
        setCurrentIndex(newInput.length);

        // Set cursor position after state update
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.setSelectionRange(newInput.length, newInput.length);
          }
        }, 0);
      }

      return;
    }

    // Store keydown time for dwell calculation
    keyDownTimes.current.set(key, timestamp);
  };

  const handleKeyUp = (e) => {
    const timestamp = Date.now();
    const key = e.key;

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
      keyCode: e.keyCode,
    };

    // Send to parent
    if (onKeystroke) {
      onKeystroke(keystrokeEvent);
    }

    // Update last keyup time
    lastKeyUp.current = timestamp;

    // Clean up
    keyDownTimes.current.delete(key);
  };

  const handleChange = (e) => {
    const newInput = e.target.value;
    const newIndex = newInput.length;

    // Update input
    setUserInput(newInput);
    setCurrentIndex(newIndex);

    // Track errors
    const newErrors = new Set();
    for (let i = 0; i < newInput.length; i++) {
      if (newInput[i] !== targetText[i]) {
        newErrors.add(i);
      }
    }
    setErrors(newErrors);
  };

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleBlur = () => {
    setIsFocused(false);
  };

  const handleContainerClick = () => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const renderText = () => {
    return targetText.split('').map((char, index) => {
      let className = 'char';

      if (index < userInput.length) {
        // Already typed
        if (errors.has(index)) {
          className += ' error';
        } else {
          className += ' correct';
        }
      } else if (index === currentIndex) {
        // Current position
        className += ' current';
      }

      // Handle special characters for display
      const isNewline = char === '\n';
      const isSpace = char === ' ';

      if (isNewline) {
        return (
          <span key={index} className={className + ' newline'}>
            ↵
            <br />
          </span>
        );
      }

      if (isSpace) {
        return (
          <span key={index} className={className + ' space'}>
            {'\u00A0'}
          </span>
        );
      }

      return (
        <span key={index} className={className}>
          {char}
        </span>
      );
    });
  };

  const accuracy = userInput.length > 0
    ? ((userInput.length - errors.size) / userInput.length * 100).toFixed(1)
    : 100;

  return (
    <div className="typing-test-container">
      <div className="typing-test-stats">
        <div className="stat">
          <span className="stat-label">Progress:</span>
          <span className="stat-value">{userInput.length} / {targetText.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Accuracy:</span>
          <span className="stat-value">{accuracy}%</span>
        </div>
        <div className="stat">
          <span className="stat-label">Errors:</span>
          <span className="stat-value">{errors.size}</span>
        </div>
      </div>

      <div
        className={`typing-test-display ${isFocused ? 'focused' : ''}`}
        onClick={handleContainerClick}
      >
        <div className="text-display">
          {renderText()}
        </div>

        {!isFocused && (
          <div className="focus-overlay">
            <span>Click here or press any key to focus</span>
          </div>
        )}
      </div>

      <textarea
        ref={inputRef}
        className="typing-test-input"
        value={userInput}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        onFocus={handleFocus}
        onBlur={handleBlur}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck="false"
        rows={1}
      />

      <div className="typing-test-hint">
        <p>Type the code exactly as shown above. Green = correct, red = error.</p>
      </div>
    </div>
  );
};

export default TypingTest;
