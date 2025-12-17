# Training the Keystroke Authentication Model

## Quick Start: Train with Your Own Keystrokes

Follow these steps to train the model with your own typing patterns:

---

## Step 1: Collect Training Data (5-10 minutes)

### 1.1 Start the Training API Server

```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation
python backend/api_training.py
```

You should see:
```
🔧 KEYSTROKE AUTHENTICATION - TRAINING MODE
Training data will be saved to: data/training_sequences.json
🚀 Server starting on http://localhost:8001
```

### 1.2 Start the Frontend

In a NEW terminal:

```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation/frontend
npm run dev
```

### 1.3 Enroll Yourself Multiple Times

1. Go to http://localhost:3000/train
2. Enter username: `hasinthad_session1`
3. Click "Start Training"
4. Complete all 4 typing exercises naturally (don't rush!)
5. **Repeat 5-10 times** with different session numbers:
   - `hasinthad_session2`
   - `hasinthad_session3`
   - ...
   - `hasinthad_session10`

**Why multiple sessions?**
- More data = better model training
- Captures natural variations in your typing
- Minimum: 3 sessions
- Recommended: 5-10 sessions
- Ideal: 10+ sessions

### 1.4 (Optional) Enroll Other Users

For even better training:
- Have friends/colleagues enroll with their names
- Each should complete 3-5 sessions
- More users = better model discrimination

---

## Step 2: Check Collected Data

```bash
curl http://localhost:8001/api/training/stats
```

You should see:
```json
{
  "total_users": 10,
  "users": {
    "hasinthad_session1": {
      "sessions": 1,
      "total_sequences": 1,
      "total_keystrokes": 250
    },
    ...
  }
}
```

**Minimum requirements for training:**
- ✅ At least 3 enrollment sessions
- ✅ At least 200 keystrokes per session
- ✅ Ideally from 2+ different users (or 10+ sessions from yourself)

---

## Step 3: Train the Model (5-10 minutes)

Stop both servers (Ctrl+C) and run:

```bash
python train_model.py --epochs 50
```

You'll see:
```
📂 Loading training data...
📊 Training Data Summary:
   👤 hasinthad_session1: ✅ Added 1 sequences
   👤 hasinthad_session2: ✅ Added 1 sequences
   ...
📦 Prepared 10 training samples from 10 users

🚀 STARTING MODEL TRAINING
⚙️  Training Configuration:
   Model: LSTM (2 layers, 64 hidden units)
   Training samples: 30
   Epochs: 50

🏋️  Training in progress...
Epoch 1/50, Loss: 0.8543
Epoch 2/50, Loss: 0.7821
...
Epoch 50/50, Loss: 0.1234

✅ TRAINING COMPLETE!
💾 Saving trained model...
   ✅ Model saved to: models/trained_keystroke_model.pth
```

**Training options:**
```bash
# Train for more epochs (better accuracy, takes longer)
python train_model.py --epochs 100

# Adjust learning rate
python train_model.py --epochs 50 --learning-rate 0.0005

# Require more sessions per user
python train_model.py --min-sessions 5
```

---

## Step 4: Test Your Trained Model

### 4.1 Update API to Use Trained Model

Edit `backend/api.py` to load the trained model:

```python
# In backend/api.py, around line 20-25
# Change from:
authenticator = KeystrokeAuthenticator()

# To:
authenticator = KeystrokeAuthenticator()
model_path = 'models/trained_keystroke_model.pth'
template_path = 'data/trained_templates.pkl'
if Path(model_path).exists():
    authenticator.load_model(model_path, template_path)
    print(f"✅ Loaded trained model from {model_path}")
```

### 4.2 Start Regular API Server

```bash
python backend/api.py
```

### 4.3 Test Recognition

1. Go to http://localhost:3000/recognize
2. Type naturally in the editor (100+ keystrokes)
3. Click "Recognize User"

**Expected results with trained model:**

| Scenario | Confidence | Level |
|----------|-----------|-------|
| Your own typing | 80-95% | HIGH 🥇 |
| Other enrolled users | 40-70% | MEDIUM/LOW 🥈 |
| Completely new user | 20-50% | LOW 🥉 |

**Without training**, results are much more random (50-70% for anyone).

---

## Step 5: Improve Accuracy

If recognition accuracy is lower than expected:

### Collect More Data
```bash
# 1. Restart training API
python backend/api_training.py

# 2. Enroll more sessions
# Go to http://localhost:3000/train
# Add 5-10 more sessions

# 3. Retrain with more data
python train_model.py --epochs 100
```

### Train for Longer
```bash
python train_model.py --epochs 100
```

### Add More Users
- Current training uses **contrastive learning**
- Model learns by comparing YOUR typing vs OTHERS
- More diverse users = better discrimination

---

## Training Data Management

### View training statistics:
```bash
curl http://localhost:8001/api/training/stats
```

### Clear training data (start fresh):
```bash
curl -X DELETE http://localhost:8001/api/training/clear
```

### View raw training data:
```bash
cat data/training_sequences.json | python -m json.tool
```

---

## Understanding the Training Process

### What happens during training?

1. **Data Collection** (`api_training.py`):
   - Saves raw keystroke sequences during enrollment
   - Stores in `data/training_sequences.json`

2. **Triplet Creation** (`train_model.py`):
   - Creates (anchor, positive, negative) triplets:
     - **Anchor**: Your typing sample
     - **Positive**: Another sample from YOU
     - **Negative**: Sample from different user
   - Goal: Make anchor similar to positive, different from negative

3. **LSTM Training**:
   - Model learns to extract typing features
   - Minimizes distance: `distance(anchor, positive) < distance(anchor, negative)`
   - Uses **Triplet Margin Loss**

4. **Embedding Learning**:
   - Trained model generates better embeddings
   - Your typing → consistent embeddings
   - Others' typing → different embeddings

### Why multiple sessions help:

- Captures natural typing variations
- Prevents overfitting to single session
- Makes model robust to different texts
- Better generalization

---

## Troubleshooting

### "No training data found"
- Make sure you ran `api_training.py` (not `api.py`)
- Check that enrollments completed successfully
- Verify file exists: `data/training_sequences.json`

### "Not enough data to create triplets"
- Need at least 2 users OR 2 sessions per user
- Collect more enrollment sessions

### "Only 1 user with enough data"
- Model will train but results limited
- Recommendation: Add more sessions or other users
- Minimum: 3 sessions from yourself
- Better: 10+ sessions or 2+ users

### Low recognition accuracy after training
- Collect more sessions (10+ recommended)
- Train for more epochs (100+)
- Ensure consistent typing style
- Add data from other users

### Model not loading in api.py
- Check paths in api.py
- Verify files exist:
  - `models/trained_keystroke_model.pth`
  - `data/trained_templates.pkl`

---

## File Structure

```
Implementation/
├── backend/
│   ├── api.py                    # Regular API (for testing)
│   ├── api_training.py           # Training API (collects data)
│   └── feature_extraction.py
├── models/
│   ├── keystroke_auth_model.py   # LSTM model
│   ├── trained_keystroke_model.pth  # Trained weights (after training)
│   └── trained_templates.pkl     # User templates (after training)
├── data/
│   ├── training_sequences.json   # Raw training data
│   └── user_templates.pkl        # Current templates
├── train_model.py                # Training script
└── TRAINING_GUIDE.md            # This file
```

---

## Quick Reference

```bash
# 1. Collect data
python backend/api_training.py
# → Enroll 5-10 times at http://localhost:3000/train

# 2. Check data
curl http://localhost:8001/api/training/stats

# 3. Train model
python train_model.py --epochs 50

# 4. Test model
python backend/api.py
# → Test at http://localhost:3000/recognize
```

---

## Expected Training Time

| Sessions | Epochs | Training Time | Quality |
|----------|--------|---------------|---------|
| 3-5      | 50     | ~2-3 min      | Basic   |
| 5-10     | 50     | ~3-5 min      | Good    |
| 10-20    | 100    | ~10-15 min    | Better  |
| 20+      | 100    | ~15-20 min    | Best    |

**Note**: Training on CPU is fast enough for this dataset size. GPU not required.

---

## Research Considerations

For academic/research use:

1. **Dataset Size**: Aim for 10+ users, 10+ sessions each
2. **Cross-validation**: Split data into train/test sets
3. **Baseline Comparison**: Test both trained vs untrained models
4. **Metrics**: Track accuracy, FAR (False Accept Rate), FRR (False Reject Rate)
5. **Documentation**: Record training parameters and results

---

## Need Help?

Check these files:
- `train_model.py` - Training script with detailed logging
- `api_training.py` - Data collection server
- `models/keystroke_auth_model.py` - Model implementation

Look for the printed messages during training for debugging info.
