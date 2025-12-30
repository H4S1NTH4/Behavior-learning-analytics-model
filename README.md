# Keystroke Dynamics Authentication System

**Behavioral Biometrics for Continuous Student Authentication**

A complete implementation of keystroke dynamics-based continuous passive authentication (CPA) for educational IDEs, based on the research paper "The Digital Proctor: A Comprehensive Analysis of Behavioral Learning Analytics and Interaction Modules in Programming Education".

## 📋 Overview

This system provides:
- ✅ **Keystroke capture** from Monaco Editor
- ✅ **Feature extraction** (dwell time, flight time, typing patterns)
- ✅ **LSTM-based authentication** model
- ✅ **Continuous monitoring** with risk scoring
- ✅ **REST API** for enrollment and verification
- ✅ **WebSocket** for real-time monitoring

## 🏗️ Architecture

```
Frontend (React + Monaco)
    ↓ WebSocket/HTTP
Backend (FastAPI)
    ↓
Feature Extraction
    ↓
LSTM Model (PyTorch)
    ↓
Authentication Decision
```

## 📁 Project Structure

```
Implementation/
├── frontend/
│   └── useKeystrokeCapture.js    # React hook for Monaco Editor
├── backend/
│   ├── api.py                     # FastAPI endpoints
│   ├── feature_extraction.py     # Keystroke feature extraction
│   └── test_system.py             # Test suite
├── models/
│   └── keystroke_auth_model.py   # LSTM authentication model
├── data/                          # Data storage (create as needed)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation
pip install -r requirements.txt
```

### 2. Run Tests

Test the system with synthetic data:

```bash
python backend/test_system.py
```

Expected output:
- ✓ Feature extraction working
- ✓ User enrollment successful
- ✓ Legitimate users authenticated
- ✓ Imposters detected

### 3. Start the API Server

```bash
python backend/api.py
```s

API will be available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 4. Integrate with Frontend

Add the React hook to your Monaco Editor component:

```javascript
import { useKeystrokeCapture } from './frontend/useKeystrokeCapture';

function MyEditor() {
  const [editor, setEditor] = useState(null);

  const handleKeystrokeData = async (batch) => {
    await fetch('http://localhost:8000/api/keystroke/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: batch })
    });
  };

  useKeystrokeCapture(
    editor,
    'student_123',  // User ID
    'session_456',  // Session ID
    handleKeystrokeData
  );

  return (
    <MonacoEditor
      onMount={(editor) => setEditor(editor)}
      // ... other props
    />
  );
}
```

## 🔧 API Usage

### Enroll a User

Collect keystroke data during practice sessions, then enroll:

```bash
curl -X POST "http://localhost:8000/api/auth/enroll" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "student_123",
    "keystrokeEvents": [...]  # Array of keystroke events
  }'
```

**Response:**
```json
{
  "success": true,
  "user_id": "student_123",
  "enrollment_complete": true,
  "message": "User enrolled successfully"
}
```

### Verify a User

```bash
curl -X POST "http://localhost:8000/api/auth/verify" \
  -H "Content-Type": application/json" \
  -d '{
    "userId": "student_123",
    "keystrokeEvents": [...],
    "threshold": 0.7
  }'
```

**Response:**
```json
{
  "success": true,
  "authenticated": true,
  "similarity": 0.85,
  "risk_score": 0.15,
  "message": "Authenticated"
}
```

### Continuous Monitoring

```bash
curl -X POST "http://localhost:8000/api/auth/monitor" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "student_123",
    "sessionId": "session_456"
  }'
```

**Response:**
```json
{
  "success": true,
  "alert_level": "LOW",
  "average_risk_score": 0.12,
  "max_risk_score": 0.25,
  "authenticated": true
}
```

## 📊 How It Works

### 1. **Data Capture** (Frontend)

Monaco Editor hooks capture:
- **Key**: Which key was pressed
- **Timestamp**: When it occurred
- **Dwell Time**: How long the key was held (keydown → keyup)
- **Flight Time**: Time between consecutive key releases

### 2. **Feature Extraction** (Backend)

From raw keystroke events, we extract:

**Static Features:**
- Mean, std, median of dwell times
- Mean, std, median of flight times
- Typing speed (chars/second)
- Error rate (backspace frequency)

**Dynamic Features:**
- Digraph timing (two-key sequences like "th", "er")
- Pause patterns
- Special key usage

### 3. **LSTM Model**

- **Input**: Sequence of keystroke features (50 keystrokes)
- **Architecture**: 2-layer LSTM → Embedding layer
- **Training**: Contrastive learning (triplet loss)
- **Output**: User-specific behavioral embedding

### 4. **Authentication**

- **Enrollment**: Create baseline template from 5+ typing samples
- **Verification**: Compare new typing to template using cosine similarity
- **Threshold**: Typically 0.5-0.7 (higher = stricter)
- **Decision**: Authenticated if similarity > threshold

### 5. **Continuous Monitoring**

- Analyze multiple recent sequences
- Calculate rolling risk score
- **Alert Levels:**
  - **LOW** (risk < 0.5): Normal behavior
  - **MEDIUM** (risk 0.5-0.7): Slight deviation
  - **HIGH** (risk > 0.7): Potential impersonation

## 📈 Model Performance

Based on research literature:

- **False Accept Rate (FAR)**: ~5% (imposters incorrectly authenticated)
- **False Reject Rate (FRR)**: ~8% (legitimate users rejected)
- **Equal Error Rate (EER)**: ~6-7%

**Factors affecting accuracy:**
- Amount of enrollment data (more = better)
- Typing context (free text vs. code vs. passwords)
- Time between enrollment and verification
- User fatigue, keyboard changes, etc.

## 🎯 Use Cases

### 1. **High-Stakes Exams**

Enable continuous monitoring during online programming exams:

```javascript
// Start monitoring when exam begins
fetch('/api/keystroke/capture', {
  method: 'POST',
  body: JSON.stringify({ events: keystrokeBuffer })
});

// Check risk score periodically
const checkAuth = setInterval(async () => {
  const result = await fetch('/api/auth/monitor', {
    method: 'POST',
    body: JSON.stringify({ userId, sessionId })
  });

  const { alert_level, risk_score } = await result.json();

  if (alert_level === 'HIGH') {
    // Trigger re-authentication or flag for review
    requireReAuthentication();
  }
}, 30000); // Every 30 seconds
```

### 2. **Practice Session Enrollment**

Collect training data during normal practice:

```javascript
// Collect data during homework/practice
const enrollmentData = [];

// After sufficient data is collected
if (enrollmentData.length > 100) {
  await fetch('/api/auth/enroll', {
    method: 'POST',
    body: JSON.stringify({
      userId: currentUser.id,
      keystrokeEvents: enrollmentData
    })
  });
}
```

### 3. **Adaptive Intervention**

Combine with struggle detection:

```javascript
// If student is struggling AND authenticated
if (strugglingDetected && riskScore < 0.3) {
  // Offer legitimate help
  showTutoringHint();
}

// If struggling AND high risk score
if (strugglingDetected && riskScore > 0.7) {
  // Possible cheating attempt
  flagForReview();
}
```

## ⚙️ Configuration

### Adjust Authentication Threshold

More lenient (fewer false rejections):
```python
verification = auth.verify_user(user_id, sequence, threshold=0.5)
```

More strict (better security):
```python
verification = auth.verify_user(user_id, sequence, threshold=0.8)
```

### Modify Sequence Length

Shorter sequences (faster decisions, less accurate):
```python
sequence = extractor.create_sequence(events, sequence_length=30)
```

Longer sequences (more accurate, slower):
```python
sequence = extractor.create_sequence(events, sequence_length=100)
```

## 🛡️ Security & Privacy

### Data Protection

- ✅ Keystroke timing (biometric) is stored, not content
- ✅ Templates are hashed embeddings, not reversible
- ✅ No actual typed text is saved
- ✅ Compliant with FERPA/GDPR principles

### Ethical Considerations

From the research paper:

**✅ DO:**
- Inform students about monitoring
- Use primarily for academic integrity in exams
- Provide transparent risk scores
- Allow opt-out for practice sessions
- Combine with human review

**❌ DON'T:**
- Use for grading effort/time
- Punish exploration/mistakes
- Fully automate high-stakes decisions
- Share biometric data with third parties

## 🔬 Research Foundation

This implementation is based on:

**Paper:** "The Digital Proctor: A Comprehensive Analysis of Behavioral Learning Analytics and Interaction Modules in Programming Education"

**Key Techniques:**
- Part II: Behavioral Biometrics (Section 2.1)
- Part III: LSTM Networks (Section 3.2)
- Part IV: Continuous Authentication Applications

**Academic References:**
- Keystroke Dynamics for authentication (1970s telegraph "Fist of Sender")
- LSTM networks for temporal sequence modeling
- Continuous Passive Authentication (CPA)

## 🐛 Troubleshooting

### Low Similarity Scores for Legitimate Users

**Causes:**
- Insufficient enrollment data
- Different typing context (code vs. text)
- Tired/different physical state

**Solutions:**
- Collect more enrollment samples (10+)
- Lower threshold temporarily
- Re-enroll periodically

### High False Accept Rate

**Causes:**
- Threshold too low
- Similar typing patterns between users
- Not enough distinctive features

**Solutions:**
- Increase threshold
- Collect longer sequences
- Add more feature types

### API Performance Issues

**Solutions:**
- Use Redis for session storage (not in-memory dict)
- Batch process keystroke events
- Use ONNX for faster inference
- Deploy model on GPU

## 📚 Next Steps

### Production Deployment

1. **Database Integration:**
   - Replace in-memory storage with TimescaleDB
   - Store keystroke events in time-series format
   - Use PostgreSQL for user profiles

2. **Model Training:**
   - Collect real student typing data (with consent)
   - Fine-tune LSTM on actual data
   - Implement online learning for template updates

3. **Monitoring Dashboard:**
   - Build instructor dashboard for alerts
   - Real-time risk score visualization
   - Session replay for forensic analysis

4. **Integration:**
   - Connect to LMS (Canvas, Moodle)
   - Integrate with existing IDE
   - Add to exam proctoring system

### Advanced Features

- **Multimodal Biometrics:** Add mouse dynamics
- **Struggle Detection:** Combine with cognitive load analysis
- **Adaptive Thresholds:** User-specific thresholds
- **Privacy Enhancement:** Homomorphic encryption

## 📞 Support

For questions about this implementation:
- Check the research PDF: `documentation/Behavioral Learning Analytics & IDE Interaction.pdf`
- Review API docs: `http://localhost:8000/docs`
- Run tests: `python backend/test_system.py`

## 📄 License

This implementation is for research and educational purposes.

---

**Built with:** Python, PyTorch, FastAPI, React, Monaco Editor

**Research-based:** Academic paper on behavioral learning analytics

**Status:** ✅ Functional prototype ready for testing
