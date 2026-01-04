# Google Colab Training Guide for TypeNet

Complete step-by-step guide to train TypeNet using your Aalto dataset in Google Colab.

---

## 📋 Prerequisites

- ✅ Processed Aalto data file: [processed_aalto_data.npy](data/processed_aalto_data.npy)
- ✅ Training script: [train_model.py](backend/train_model.py)
- ✅ Google account with Google Drive access
- ✅ Google Colab (free tier works, but GPU recommended)

---

## 🚀 Step-by-Step Instructions

### Step 1: Prepare Your Google Drive

1. **Upload your data file to Google Drive**:
   - Go to [Google Drive](https://drive.google.com)
   - Upload `processed_aalto_data.npy` to your Drive root (My Drive)
   - Path should be: `My Drive/processed_aalto_data.npy`

2. **Verify data format**:
   Your data should have shape: `(Num_Users, 5_Sequences, 70, 5)`
   - Num_Users: Number of users in dataset
   - 5_Sequences: 5 typing sequences per user
   - 70: Sequence length (70 keystrokes)
   - 5: Features [HL, IL, PL, RL, KeyCode]

---

### Step 2: Create New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click "New Notebook"
3. Name it: "TypeNet_Training"

---

### Step 3: Enable GPU (Recommended)

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Click **Save**

---

### Step 4: Mount Google Drive

**Cell 1:** Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

Run this cell and authorize access when prompted.

---

### Step 5: Verify Data File

**Cell 2:** Check data
```python
import numpy as np
import os

# Check if file exists
data_path = '/content/drive/My Drive/processed_aalto_data.npy'
if os.path.exists(data_path):
    print(f"✅ Data file found!")

    # Load and check shape
    data = np.load(data_path, allow_pickle=True)
    print(f"📊 Data shape: {data.shape}")
    print(f"   - Number of users: {data.shape[0]}")
    print(f"   - Sequences per user: {data.shape[1]}")
    print(f"   - Keystrokes per sequence: {data.shape[2]}")
    print(f"   - Features per keystroke: {data.shape[3]}")

    # Expected: (Num_Users, 5, 70, 5)
    if data.shape[2] == 70 and data.shape[3] == 5:
        print("✅ Data format is correct!")
    else:
        print(f"⚠️ Warning: Expected shape (*, 5, 70, 5), got {data.shape}")
else:
    print(f"❌ Data file not found at: {data_path}")
    print("Please upload 'processed_aalto_data.npy' to your Google Drive root folder")
```

---

### Step 6: Install Dependencies

**Cell 3:** Install PyTorch (usually pre-installed in Colab)
```python
# Check PyTorch installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# If CUDA available, show GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### Step 7: Upload Training Script

**Option A: Copy-paste the code directly**

**Cell 4:** TypeNet Training Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- CONFIGURATION (Based on TypeNet Research) ---
DATA_PATH = '/content/drive/My Drive/processed_aalto_data.npy'
MODEL_SAVE_PATH = '/content/drive/My Drive/typenet_pretrained.pth'

# Hyperparameters from the papers
INPUT_SIZE = 5       # HL, IL, PL, RL, KeyCode
HIDDEN_SIZE = 128    # 128 units per LSTM layer
NUM_LAYERS = 2       # 2 stacked LSTM layers
OUTPUT_SIZE = 128    # Embedding dimension
DROPOUT_RATE = 0.5   # Dropout between LSTM layers
SEQUENCE_LENGTH = 70 # Optimal sequence length
BATCH_SIZE = 512     # Large batch size for stable triplet loss
LEARNING_RATE = 0.005 # Learning rate (tuned for Adam)
MARGIN = 1.5         # Triplet Loss margin
EPOCHS = 100         # Sufficient for convergence

# --- 1. THE DATASET (Triplet Sampling) ---
class KeystrokeTripletDataset(Dataset):
    def __init__(self, npy_path):
        print(f"Loading data from {npy_path}...")
        self.data = np.load(npy_path, allow_pickle=True)
        self.num_users = self.data.shape[0]
        self.num_sequences = self.data.shape[1]
        print(f"Data Loaded. Users: {self.num_users}, Seq/User: {self.num_sequences}")

    def __len__(self):
        return self.num_users * 10

    def __getitem__(self, index):
        # 1. Select Anchor User
        anchor_user_idx = index % self.num_users

        # 2. Select Positive Sample (Same User, different sequence)
        seq_indices = np.random.choice(self.num_sequences, size=2, replace=False)
        anchor_seq = self.data[anchor_user_idx, seq_indices[0]]
        positive_seq = self.data[anchor_user_idx, seq_indices[1]]

        # 3. Select Negative Sample (Different User)
        negative_user_idx = np.random.randint(0, self.num_users)
        while negative_user_idx == anchor_user_idx:
            negative_user_idx = np.random.randint(0, self.num_users)

        negative_seq_idx = np.random.randint(0, self.num_sequences)
        negative_seq = self.data[negative_user_idx, negative_seq_idx]

        return (torch.from_numpy(anchor_seq),
                torch.from_numpy(positive_seq),
                torch.from_numpy(negative_seq))

# --- 2. THE MODEL (TypeNet Architecture) ---
class TypeNet(nn.Module):
    def __init__(self):
        super(TypeNet, self).__init__()

        # LSTM Layer 1
        self.lstm1 = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.bn1 = nn.BatchNorm1d(HIDDEN_SIZE)  # Batch Norm across hidden dimension
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        # LSTM Layer 2
        self.lstm2 = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.bn2 = nn.BatchNorm1d(HIDDEN_SIZE)  # Batch Norm across hidden dimension
        self.dropout2 = nn.Dropout(DROPOUT_RATE)

        # Output Embedding Layer (Dense)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward_one(self, x):
        # Pass through LSTM 1
        out, _ = self.lstm1(x)
        out = out.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)
        out = self.dropout1(out)

        # Pass through LSTM 2
        out, _ = self.lstm2(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = out.permute(0, 2, 1)
        out = self.dropout2(out)

        # Take the output of the LAST timestep
        last_timestep = out[:, -1, :]

        # Final Embedding
        embedding = self.fc(last_timestep)
        return embedding

    def forward(self, anchor, positive, negative):
        emb_a = self.forward_one(anchor)
        emb_p = self.forward_one(positive)
        emb_n = self.forward_one(negative)
        return emb_a, emb_p, emb_n

# --- 3. THE LOSS FUNCTION (Triplet Loss) ---
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = torch.pow(anchor - positive, 2).sum(dim=1)
        dist_neg = torch.pow(anchor - negative, 2).sum(dim=1)
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()

# --- 4. TRAINING LOOP ---
def train_typenet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")

    # Load Data
    dataset = KeystrokeTripletDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize Model
    model = TypeNet().to(device)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🏋️ Starting Training Loop...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            # Move to GPU
            anchor = anchor.to(device).float()
            positive = positive.to(device).float()
            negative = negative.to(device).float()

            # Forward Pass
            optimizer.zero_grad()
            emb_a, emb_p, emb_n = model(anchor, positive, negative)

            # Compute Loss
            loss = criterion(emb_a, emb_p, emb_n)

            # Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] Complete. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Model saved to {MODEL_SAVE_PATH}")

    print("🎉 Training Complete!")
    return model

# Run training
model = train_typenet()
```

**Option B: Upload train_model.py file**

1. Click the **Files** icon (folder) in left sidebar
2. Click **Upload** button
3. Select `backend/train_model.py` from your computer
4. Then run:

```python
!python train_model.py
```

---

### Step 8: Monitor Training

Training will output:
```
🚀 Training on device: cuda
Loading data from /content/drive/My Drive/processed_aalto_data.npy...
Data Loaded. Users: 136, Seq/User: 5
🏋️ Starting Training Loop...
Epoch 1 | Batch 0 | Loss: 1.5234
Epoch 1 | Batch 10 | Loss: 1.4876
...
✅ Epoch [1/100] Complete. Avg Loss: 1.4532
💾 Model saved to /content/drive/My Drive/typenet_pretrained.pth
```

**Training Time Estimates:**
- With GPU (T4): ~2-4 hours for 100 epochs
- With CPU: ~10-20 hours (not recommended)

---

### Step 9: Download Trained Model

**Option A: Download directly from Colab**

**Cell 5:** Download model
```python
from google.colab import files

# Model should be saved in Google Drive
model_path = '/content/drive/My Drive/typenet_pretrained.pth'

if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
    print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

    # Download to your computer
    files.download(model_path)
    print("✅ Download started!")
else:
    print(f"❌ Model not found at: {model_path}")
```

**Option B: Download from Google Drive**
1. Go to Google Drive
2. Find `typenet_pretrained.pth` in "My Drive"
3. Right-click → Download

---

### Step 10: Verify Trained Model

**Cell 6:** Test model
```python
import torch

# Load model
model = TypeNet()
model.load_state_dict(torch.load('/content/drive/My Drive/typenet_pretrained.pth', map_location='cpu'))
model.eval()

print("✅ Model loaded successfully!")

# Test with random data
test_input = torch.randn(1, 70, 5)  # (batch=1, seq_len=70, features=5)
with torch.no_grad():
    embedding = model.forward_one(test_input)
    print(f"✅ Output embedding shape: {embedding.shape}")
    print(f"✅ Expected shape: torch.Size([1, 128])")

if embedding.shape == torch.Size([1, 128]):
    print("🎉 Model is working correctly!")
```

---

## 📊 Monitoring Training Progress

### View Loss Curves

Add this to monitor training:

```python
import matplotlib.pyplot as plt

# After training, plot loss
def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TypeNet Training Loss')
    plt.grid(True)
    plt.show()
```

### Check GPU Usage

```python
# Monitor GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
```

---

## ⚙️ Configuration Options

### Adjust Hyperparameters

If training is too slow or memory issues:

```python
# Reduce batch size (if out of memory)
BATCH_SIZE = 256  # Default: 512

# Reduce epochs (for quick testing)
EPOCHS = 50  # Default: 100

# Adjust learning rate
LEARNING_RATE = 0.001  # Default: 0.005
```

### Save Checkpoints More Frequently

```python
# In training loop, change:
if (epoch + 1) % 5 == 0:  # Save every 5 epochs instead of 10
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
**Solution**: Reduce batch size
```python
BATCH_SIZE = 128  # or even 64
```

### Error: "File not found"
**Solution**: Check file path
```python
# List files in Google Drive
!ls "/content/drive/My Drive/"
```

### Error: "Runtime disconnected"
**Solution**:
- Colab free tier has usage limits (~12 hours)
- Save checkpoints frequently
- Consider Colab Pro for longer sessions

### Training is very slow
**Solutions**:
- Make sure GPU is enabled (Runtime → Change runtime type)
- Reduce number of epochs for testing
- Use smaller dataset for quick validation

---

## 📥 After Training

Once training completes and you've downloaded `typenet_pretrained.pth`:

1. **Place in your project**:
   ```bash
   # Copy to models folder
   cp typenet_pretrained.pth /path/to/Implementation/models/
   ```

2. **Test locally**:
   ```bash
   python test_typenet.py
   ```

3. **Start backend**:
   ```bash
   python backend/api.py
   ```

4. **Verify integration**:
   ```bash
   python test_backend_typenet.py
   ```

---

## 📚 Additional Tips

### Save Training Logs

```python
# Add to training loop
with open('/content/drive/My Drive/training_log.txt', 'a') as f:
    f.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}\n")
```

### Resume Training from Checkpoint

```python
# Load existing model
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("✅ Resuming from checkpoint")
```

### Use TensorBoard (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/content/drive/My Drive/runs')
# In training loop:
writer.add_scalar('Loss/train', avg_loss, epoch)
```

---

## ✅ Checklist

- [ ] Upload `processed_aalto_data.npy` to Google Drive
- [ ] Create Colab notebook
- [ ] Enable GPU runtime
- [ ] Mount Google Drive
- [ ] Verify data format
- [ ] Copy training code
- [ ] Start training
- [ ] Monitor progress
- [ ] Download trained model
- [ ] Test model locally

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify data format and paths
3. Make sure GPU is enabled
4. Try reducing batch size
5. Check Google Drive has enough space (~200MB for model)

---

**Estimated Total Time**: 3-5 hours (mostly training time)

**Expected Model File Size**: ~100-200 MB

**Next Step**: See [TYPENET_INTEGRATION_SUMMARY.md](TYPENET_INTEGRATION_SUMMARY.md) for deployment instructions.
