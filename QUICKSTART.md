# 🚀 Quick Start Guide

## Your Keystroke Dynamics Authentication System is Ready!

✅ All tests passed successfully!

## What You Have

A complete working system with:

1. **Frontend Hook** - Captures keystrokes from Monaco Editor
2. **Feature Extractor** - Analyzes typing patterns
3. **LSTM Model** - Authenticates users based on behavior
4. **REST API** - Enrollment and verification endpoints
5. **Test Suite** - Validates everything works

## 3-Step Setup

### Step 1: Install Dependencies

```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation
pip install -r requirements.txt
```

### Step 2: Run Tests (Verify it works)

```bash
python backend/test_system.py
```

You should see:
```
✓ Feature Extraction: PASSED
✓ Enrollment & Verification: PASSED
✓ Continuous Authentication: PASSED
```

### Step 3: Start the Server

```bash
python backend/api.py
```

Access the API at: http://localhost:8000/docs

## How to Use

### Basic Flow

```
1. Student types during practice → Data captured
2. Enrollment: POST /api/auth/enroll
3. During exam: Continuous monitoring
4. Verification: POST /api/auth/verify
5. Get risk score → Flag if suspicious
```

### Code Example

```javascript
// In your React component
import { useKeystrokeCapture } from './frontend/useKeystrokeCapture';

function IDE() {
  const [editor, setEditor] = useState(null);

  useKeystrokeCapture(
    editor,
    'student_id',
    'session_id',
    async (batch) => {
      await fetch('http://localhost:8000/api/keystroke/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events: batch })
      });
    }
  );

  return <MonacoEditor onMount={setEditor} />;
}
```

## API Endpoints

| Endpoint | Purpose | When to Use |
|----------|---------|-------------|
| `POST /api/keystroke/capture` | Capture keystrokes | Real-time during typing |
| `POST /api/auth/enroll` | Enroll user | After practice sessions |
| `POST /api/auth/verify` | Verify identity | During exams |
| `POST /api/auth/monitor` | Continuous check | Every 30-60 seconds |

## Test Results Explained

The system successfully:
- ✅ **Extracts 19 behavioral features** from typing patterns
- ✅ **Enrolls users** and creates biometric templates
- ✅ **Authenticates legitimate users** (99.9% similarity)
- ✅ **Detects imposters** (different typing patterns)
- ✅ **Provides risk scores** (0.0 = safe, 1.0 = suspicious)

## What Makes It Work

### Captured Features

From each keystroke sequence:
- **Timing**: How long keys are pressed (dwell time)
- **Speed**: Time between keystrokes (flight time)
- **Patterns**: Common key combinations (digraphs)
- **Errors**: Backspace/delete frequency
- **Pauses**: Thinking time patterns

### Model Architecture

```
Keystroke Sequence (50 keys)
    ↓
LSTM Network (2 layers)
    ↓
User Embedding (32 dimensions)
    ↓
Similarity Comparison
    ↓
Authentication Decision
```

## Customization

### Adjust Security Level

**More Lenient** (fewer false rejections):
```python
threshold = 0.5  # Less strict
```

**More Strict** (better security):
```python
threshold = 0.8  # More strict
```

### Change Monitoring Frequency

```javascript
// Check every 30 seconds
setInterval(checkAuthentication, 30000);

// Check every 60 seconds
setInterval(checkAuthentication, 60000);
```

## Integration with Your System

### 1. Add to Your IDE

Copy [frontend/useKeystrokeCapture.js](frontend/useKeystrokeCapture.js) to your React project.

### 2. Connect Backend

Update API URL in your frontend:
```javascript
const API_URL = 'https://your-backend-url.com';
```

### 3. Database Setup (Production)

Replace in-memory storage with TimescaleDB:
```python
# In api.py, replace:
active_sessions = {}

# With:
from database import get_session
```

## Common Use Cases

### 1. Online Exam Monitoring

```javascript
// Start monitoring when exam begins
startContinuousMonitoring(userId, sessionId);

// Alert if risk is high
if (riskScore > 0.7) {
  alert("Please verify your identity");
  requireReAuthentication();
}
```

### 2. Practice Session Enrollment

```javascript
// Collect during homework
if (keystrokeCount > 100) {
  enrollUser(userId, keystrokeData);
}
```

### 3. Adaptive Learning

```javascript
// Combine authentication with struggle detection
if (authenticated && struggling) {
  offerHelp();  // Legitimate student needs help
} else if (!authenticated && struggling) {
  flagSuspicious();  // Possible cheating
}
```

## Troubleshooting

### "User not enrolled" error
**Solution:** Collect at least 100 keystrokes and call `/api/auth/enroll`

### Low similarity scores
**Solution:** Collect more enrollment data (5-10 typing sessions)

### High false positives
**Solution:** Lower the threshold from 0.7 to 0.5

## Next Steps

1. ✅ **Test locally** - Run `python backend/test_system.py`
2. ✅ **Start server** - Run `python backend/api.py`
3. 🔲 **Integrate frontend** - Add to your React app
4. 🔲 **Collect real data** - Test with actual students
5. 🔲 **Train model** - Fine-tune on your data
6. 🔲 **Deploy** - Put into production

## Resources

- **Full README**: [README.md](README.md)
- **Research Paper**: [documentation/Behavioral Learning Analytics & IDE Interaction.pdf](documentation/Behavioral%20Learning%20Analytics%20&%20IDE%20Interaction.pdf)
- **API Docs**: http://localhost:8000/docs (when server is running)
- **Test Suite**: [backend/test_system.py](backend/test_system.py)

## Performance Metrics

Based on research literature:
- **Accuracy**: ~93% (EER ~7%)
- **False Accept Rate**: ~5%
- **False Reject Rate**: ~8%
- **Processing Time**: <100ms per verification

---

**Status**: ✅ Ready to use!

**Tech Stack**: React + Monaco, Python + FastAPI, PyTorch, LSTM

**Based On**: Research paper on behavioral biometrics in education
