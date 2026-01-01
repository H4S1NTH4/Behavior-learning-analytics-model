"""
Model Training Script for Keystroke Biometrics

This script loads the raw keystroke sequences collected via `api_training.py`,
trains a Siamese LSTM network to learn user-specific typing patterns, and saves
the trained model for inference.

Run this script after collecting sufficient data:
    python backend/train_model.py
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuration ---
TRAINING_DATA_PATH = Path(__file__).parent.parent / 'data' / 'training_sequences.json'
MODEL_SAVE_PATH = Path(__file__).parent.parent / 'models' / 'keystroke_biometric_model.pth'

# Hyperparameters
INPUT_SIZE = 8  # Corresponds to the number of features per keystroke event
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 32 # Embedding size
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 16

# --- 1. Model Definition (Siamese LSTM) ---
class SiameseLSTM(nn.Module):
    def __init__(self):
        super(SiameseLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, OUTPUT_SIZE) # Bidirectional so * 2

    def forward_once(self, x):
        # LSTM returns output, (hidden_state, cell_state)
        _, (hidden, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# --- 2. Contrastive Loss Function ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 3. Data Loading and Preparation ---
class KeystrokeTripletDataset(Dataset):
    def __init__(self, data, user_map):
        self.data = data
        self.user_map = user_map
        self.users = list(user_map.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_seq, anchor_user_id = self.data[index]

        # Decide whether to pull a positive or negative sample
        should_get_positive = np.random.random() < 0.5
        
        if should_get_positive:
            # Positive pair (same user)
            positive_list = [item for item in self.data if item[1] == anchor_user_id and item != (anchor_seq, anchor_user_id)]
            positive_item = positive_list[np.random.randint(len(positive_list))]
            return torch.FloatTensor(anchor_seq), torch.FloatTensor(positive_item[0]), torch.FloatTensor([0.0])
        else:
            # Negative pair (different user)
            negative_user_id = anchor_user_id
            while negative_user_id == anchor_user_id:
                negative_user_id = np.random.choice(self.users)
            
            negative_list = [item for item in self.data if item[1] == negative_user_id]
            negative_item = negative_list[np.random.randint(len(negative_list))]
            return torch.FloatTensor(anchor_seq), torch.FloatTensor(negative_item[0]), torch.FloatTensor([1.0])

def load_data():
    """Loads sequences from the JSON file."""
    with open(TRAINING_DATA_PATH, 'r') as f:
        training_data = json.load(f)

    all_sequences = []
    user_map = {user_id: i for i, user_id in enumerate(training_data.keys())}

    for user_id, data in training_data.items():
        for session in data['sessions']:
            # The saved sequence might be a list of lists, ensure it's a valid sequence
            if 'sequence' in session and len(session['sequence']) > 0:
                 all_sequences.append((session['sequence'], user_id))

    return all_sequences, user_map

# --- 4. Training Loop ---
if __name__ == "__main__":
    print("🚀 Starting Keystroke Biometric Model Training...")

    # Load data
    sequences, user_map = load_data()
    if len(user_map) < 2:
        print("❌ Error: Need data from at least 2 users to train. Exiting.")
        exit()

    print(f"✅ Loaded {len(sequences)} sequences from {len(user_map)} users.")

    # Create dataset and dataloader
    train_dataset = KeystrokeTripletDataset(sequences, user_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SiameseLSTM()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("🏋️ Training model...")
    for epoch in range(NUM_EPOCHS):
        for i, (anchor, pair, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output1, output2 = model(anchor, pair)
            loss = criterion(output1, output2, label.squeeze())
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    # Save the trained model's state dictionary
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    # We only need the core LSTM and FC layers for creating templates later
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Training complete. Model saved to: {MODEL_SAVE_PATH}")