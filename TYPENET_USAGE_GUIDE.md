# TypeNet Usage Guide

## 🎯 Complete Workflow: Training to Deployment

### Step 1: Train TypeNet in Google Colab

1. **Upload `train_model.py` to Colab**
2. **Mount Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Ensure your data file exists**:
```python
# Your data should be at: /content/drive/My Drive/processed_aalto_data.npy
# Shape: (Num_Users, 5_Sequences, 70, 5)
import numpy as np
data = np.load('/content/drive/My Drive/processed_aalto_data.npy')
print(f"Data shape: {data.shape}")
```

4. **Run training**:
```python
!python train_model.py
```

5. **Download trained model**:
```python
from google.colab import files
files.download('/content/drive/My Drive/typenet_pretrained.pth')
```

---

### Step 2: Use Trained Model Locally

#### Option A: Quick Test Script

```python
from models.typenet_inference import TypeNetAuthenticator
import numpy as np

# Initialize with your trained model
auth = TypeNetAuthenticator(
    model_path='models/typenet_pretrained.pth',
    device='cpu'  # or 'cuda' if you have GPU
)

# Enroll a user (need at least 3-5 sequences)
user_id = "student_001"

# Your keystroke data: (70, 5) shape
# Features: [Hold_Time, Inter-key_Latency, Press_Latency, Release_Latency, KeyCode]
enrollment_sequences = [
    np.array([...]),  # 70x5 array - sequence 1
    np.array([...]),  # 70x5 array - sequence 2
    np.array([...]),  # 70x5 array - sequence 3
    np.array([...]),  # 70x5 array - sequence 4
    np.array([...]),  # 70x5 array - sequence 5
]

# Enroll user
result = auth.enroll_user(user_id, enrollment_sequences)
print("Enrollment:", result)

# Verify user with new sequence
test_sequence = np.array([...])  # 70x5 array
verification = auth.verify_user(user_id, test_sequence, threshold=0.7)
print("Verification:", verification)

# Identify unknown user
identification = auth.identify_user(test_sequence, top_k=3)
print("Identification:", identification)

# Save user templates
auth.save_templates('models/user_templates.pkl')
```

---

### Step 3: Integrate with Backend API

#### Update your Flask/FastAPI backend:

```python
# backend/app.py
from models.typenet_inference import TypeNetAuthenticator
import numpy as np

# Initialize once at startup
typenet_auth = TypeNetAuthenticator(
    model_path='models/typenet_pretrained.pth',
    device='cpu'
)

# Try to load existing templates
try:
    typenet_auth.load_templates('models/user_templates.pkl')
except:
    print("No existing templates found")

@app.post("/api/auth/enroll")
def enroll_user(request):
    user_id = request.json['userId']
    sequences = request.json['sequences']  # List of sequences

    # Convert to numpy arrays
    sequences_np = [np.array(seq) for seq in sequences]

    # Enroll
    result = typenet_auth.enroll_user(user_id, sequences_np)

    # Save templates
    typenet_auth.save_templates('models/user_templates.pkl')

    return result

@app.post("/api/auth/verify")
def verify_user(request):
    user_id = request.json['userId']
    sequence = np.array(request.json['sequence'])

    result = typenet_auth.verify_user(user_id, sequence, threshold=0.7)
    return result

@app.post("/api/auth/identify")
def identify_user(request):
    sequence = np.array(request.json['sequence'])

    result = typenet_auth.identify_user(sequence, top_k=3)
    return result
```

---

## 🔑 Key Requirements

### Input Data Format:
- **Shape**: `(70, 5)` - Exactly 70 keystrokes, 5 features each
- **Features**:
  1. Hold Time (HL)
  2. Inter-key Latency (IL)
  3. Press Latency (PL)
  4. Release Latency (RL)
  5. Key Code

### Example keystroke sequence:
```python
sequence = np.array([
    [0.123, 0.056, 0.089, 0.034, 65],  # Keystroke 1: 'A' key
    [0.145, 0.078, 0.091, 0.047, 83],  # Keystroke 2: 'S' key
    # ... 68 more keystrokes ...
])
# Shape: (70, 5)
```

---

## 📊 Model Performance Thresholds

Based on TypeNet research:

| Threshold | Security Level | Use Case |
|-----------|---------------|----------|
| 0.9 | Very High | Banking, sensitive data |
| 0.8 | High | Corporate login |
| 0.7 | Medium | General authentication |
| 0.6 | Low | Continuous monitoring |

---

## ⚠️ Important Notes

1. **Sequence Length**: Must be exactly 70 keystrokes (as per training)
   - If you have shorter sequences, you'll need to:
     - Pad them to 70
     - Or retrain with different `SEQUENCE_LENGTH`

2. **Feature Extraction**: Make sure your frontend captures all 5 features correctly

3. **Enrollment**: Collect 5-10 enrollment samples per user for best results

4. **Model Location**: Place `typenet_pretrained.pth` in `models/` directory

5. **Templates**: Save user templates regularly to persist enrollments

---

## 🧪 Testing Your Setup

```python
# test_typenet.py
from models.typenet_inference import TypeNetAuthenticator
import numpy as np

# Test 1: Load model
print("Test 1: Loading model...")
auth = TypeNetAuthenticator(
    model_path='models/typenet_pretrained.pth'
)
print("✅ Model loaded successfully")

# Test 2: Generate embedding
print("\nTest 2: Generating embedding...")
test_seq = np.random.randn(70, 5)
embedding = auth.get_embedding(test_seq)
print(f"✅ Embedding shape: {embedding.shape}")

# Test 3: Enroll and verify
print("\nTest 3: Enroll and verify...")
user_sequences = [np.random.randn(70, 5) for _ in range(5)]
enroll_result = auth.enroll_user("test_user", user_sequences)
print(f"✅ Enrollment: {enroll_result}")

verify_result = auth.verify_user("test_user", np.random.randn(70, 5))
print(f"✅ Verification: {verify_result}")

print("\n🎉 All tests passed!")
```

---

## 🚀 Next Steps

1. ✅ Train TypeNet in Colab
2. ✅ Download trained model
3. ✅ Test with `typenet_inference.py`
4. 🔄 Integrate with your backend
5. 🔄 Connect to frontend keystroke capture
6. 🔄 Test with real user data
7. 🔄 Deploy to production

---

## 🆚 Comparison: TypeNet vs Simple LSTM

| Feature | TypeNet (trained) | keystroke_auth_model.py |
|---------|------------------|------------------------|
| Input features | 5 (HL, IL, PL, RL, KeyCode) | 3 (dwell, flight, category) |
| Hidden size | 128 | 64 |
| Embedding size | 128 | 32 |
| Pre-trained | Yes (on large dataset) | No |
| Performance | Higher (research-backed) | Lower (untrained) |
| Use this for | Production system | Prototyping only |

**Recommendation**: Use TypeNet (typenet_inference.py) for your final system!
