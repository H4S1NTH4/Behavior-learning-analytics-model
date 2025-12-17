# Quick Start: Train with Your Own Keystrokes

## TL;DR

```bash
# 1. Collect training data (5-10 minutes)
python backend/api_training.py
# → Open http://localhost:3000/train
# → Enroll yourself 5-10 times (hasinthad_session1, hasinthad_session2, etc.)

# 2. Train the model (5 minutes)
python train_model.py --epochs 50

# 3. Test it
python backend/api.py
# → Open http://localhost:3000/recognize
# → Type and click "Recognize User"
```

---

## Detailed Steps

### Step 1: Collect Your Typing Data

**Terminal 1 - Start training server:**
```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation
python backend/api_training.py
```

**Terminal 2 - Start frontend:**
```bash
cd /home/hasinthad/Documents/SLIIT/Research/Implementation/frontend
npm run dev
```

**Browser:**
1. Go to http://localhost:3000/train
2. Enter username: `hasinthad_session1`
3. Complete all 4 typing exercises
4. **Repeat 5-10 times** with different session numbers:
   - `hasinthad_session2`
   - `hasinthad_session3`
   - ... (continue to session 10)

**Why 5-10 times?**
- More data = better accuracy
- Captures your natural typing variations
- Each session takes ~3 minutes

---

### Step 2: Train the Model

Stop both servers (Ctrl+C), then:

```bash
python train_model.py --epochs 50
```

Wait 5 minutes while it trains. You'll see:
```
📦 Prepared 10 training samples from 10 users
🏋️  Training in progress...
Epoch 1/50, Loss: 0.8543
...
Epoch 50/50, Loss: 0.1234
✅ TRAINING COMPLETE!
💾 Saving trained model...
```

---

### Step 3: Test Recognition

Start the regular server:

```bash
python backend/api.py
```

You should see:
```
✅ Loaded trained model from: models/trained_keystroke_model.pth
```

**Test it:**
1. Open http://localhost:3000/recognize
2. Type naturally (100+ keystrokes)
3. Click "Recognize User"
4. Check if it identifies you correctly!

**Expected results:**
- Your typing: **80-95% confidence (HIGH)** 🥇
- Other users: **40-70% confidence (MEDIUM/LOW)** 🥈

---

## What Gets Saved

After training, you'll have:

```
Implementation/
├── data/
│   ├── training_sequences.json      # Your raw keystroke data
│   └── trained_templates.pkl        # User profiles
├── models/
│   └── trained_keystroke_model.pth  # Trained LSTM weights
```

The regular `api.py` automatically loads the trained model if it exists.

---

## Improving Accuracy

Not getting good results? Try:

### 1. More Training Data
```bash
# Collect 10-20 more sessions
python backend/api_training.py
# → Enroll more times with different session numbers
```

### 2. Longer Training
```bash
python train_model.py --epochs 100
```

### 3. Add More Users
- Have friends enroll (3-5 sessions each)
- Model learns by comparing YOU vs OTHERS
- More diversity = better discrimination

---

## Troubleshooting

**"No training data found"**
- Make sure you used `api_training.py` (not `api.py`) during data collection

**"Model not loading"**
- Check if files exist:
  - `models/trained_keystroke_model.pth`
  - `data/trained_templates.pkl`
- If missing, retrain: `python train_model.py`

**Low accuracy even after training**
- Collect more sessions (10+ recommended)
- Train longer (--epochs 100)
- Type consistently (don't switch between styles)

---

## Full Documentation

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive details.

---

## Quick Commands Reference

```bash
# Collect data
python backend/api_training.py

# Check collected data
curl http://localhost:8001/api/training/stats

# Train model
python train_model.py --epochs 50

# Test model
python backend/api.py

# Clear training data (start fresh)
curl -X DELETE http://localhost:8001/api/training/clear
```
