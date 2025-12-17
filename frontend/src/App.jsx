/**
 * Main Application Component
 * Provides navigation and routing for Train and Recognize pages
 */

import React from 'react';
import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';
import { FaKeyboard } from 'react-icons/fa';
import TrainUser from './components/TrainUser';
import RecognizeUser from './components/RecognizeUser';
import './App.css';

const App = () => {
  return (
    <BrowserRouter>
      <div className="app">
        {/* Navigation Bar */}
        <nav className="app-nav">
          <div className="nav-brand">
            <FaKeyboard className="brand-icon" />
            <span>Keystroke Authentication</span>
          </div>
          <div className="nav-links">
            <Link to="/train" className="nav-link">
              Train Users
            </Link>
            <Link to="/recognize" className="nav-link">
              Recognize User
            </Link>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Navigate to="/train" replace />} />
          <Route path="/train" element={<TrainUser />} />
          <Route path="/recognize" element={<RecognizeUser />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
};

export default App;
