"""
Keystroke Authentication Model Training Script
Trains the LSTM model using collected keystroke data

Usage:
    python train_model.py --epochs 50 --batch-size 32
"""

import sys
import argparse
import numpy as np
import json
from pathlib import Path
import torch

# Add paths
sys.path.append(str(Path(__file__).parent))

from models.keystroke_auth_model import KeystrokeAuthenticator


def load_training_sequences():
    """Load raw keystroke sequences from training data"""
    data_path = Path(__file__).parent / 'data' / 'training_sequences.json'

    if not data_path.exists():
        print()
        print("❌ ERROR: No training data found!")
        print(f"   Expected file: {data_path}")
        print()
        print("📋 To collect training data:")
        print()
        print("   1. Run the training API server:")
        print("      python backend/api_training.py")
        print()
        print("   2. Start the frontend:")
        print("      cd frontend && npm run dev")
        print()
        print("   3. Enroll yourself multiple times:")
        print("      - Go to http://localhost:3000/train")
        print("      - Enroll with: your_name_session1")
        print("      - Complete all 4 exercises")
        print("      - Repeat 5-10 times with different session numbers")
        print()
        print("   4. (Recommended) Enroll other users too")
        print()
        sys.exit(1)

    print("📂 Loading training data...")
    with open(data_path, 'r') as f:
        training_data = json.load(f)

    return training_data


def prepare_training_pairs(training_data, min_sessions=3):
    """
    Prepare training pairs from collected data
    Returns list of (sequence, user_id) tuples
    """
    print()
    print("📊 Training Data Summary:")
    print()

    training_pairs = []
    valid_users = []

    for user_id, user_data in training_data.items():
        num_sessions = len(user_data['sessions'])
        total_keystrokes = sum(s['num_keystrokes'] for s in user_data['sessions'])

        print(f"   👤 {user_id}:")
        print(f"      Sessions: {num_sessions}")
        print(f"      Total keystrokes: {total_keystrokes}")

        if num_sessions < min_sessions:
            print(f"      ⚠️  Skipped (need at least {min_sessions} sessions)")
            continue

        # Add all sequences for this user
        for session in user_data['sessions']:
            sequence = np.array(session['sequence'])
            training_pairs.append((sequence, user_id))

        valid_users.append(user_id)
        print(f"      ✅ Added {num_sessions} sequences")

    print()
    print(f"📦 Prepared {len(training_pairs)} training samples from {len(valid_users)} users")

    if len(valid_users) < 2:
        print()
        print("⚠️  WARNING: Only 1 user with enough data!")
        print("   For effective training, you should have:")
        print("   - Multiple sessions from yourself (5-10+), AND/OR")
        print("   - Multiple different users (2-5+)")
        print()
        print("   Training will continue but results may be limited.")
        print()

    return training_pairs, valid_users


def create_triplet_batches(training_pairs, batch_size=32):
    """
    Create triplet batches for contrastive learning
    Each batch contains (anchor, positive, negative) triplets
    """
    print()
    print("🔄 Creating triplet batches...")

    # Organize by user
    user_samples = {}
    for sequence, user_id in training_pairs:
        if user_id not in user_samples:
            user_samples[user_id] = []
        user_samples[user_id].append(sequence)

    user_ids = list(user_samples.keys())
    triplet_data = []

    # Create balanced triplets
    for user_id in user_ids:
        samples = user_samples[user_id]

        # Need at least 2 samples per user for anchor-positive pairs
        if len(samples) < 2:
            continue

        # For each sample, create triplets
        for i in range(len(samples)):
            # Anchor
            anchor_seq = samples[i]

            # Positive (different sample from same user)
            pos_idx = (i + 1) % len(samples)
            positive_seq = samples[pos_idx]

            # Negative (sample from different user)
            if len(user_ids) > 1:
                other_users = [u for u in user_ids if u != user_id]
                neg_user = np.random.choice(other_users)
                negative_seq = np.random.choice(user_samples[neg_user])

                # Add triplet
                triplet_data.append((anchor_seq, user_id))
                triplet_data.append((positive_seq, user_id))
                triplet_data.append((negative_seq, neg_user))

    print(f"✅ Created {len(triplet_data)} triplet samples")

    return triplet_data


def train_model(training_pairs, epochs=50, learning_rate=0.001):
    """
    Train the LSTM model using triplet loss
    """
    print()
    print("=" * 70)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 70)
    print()

    # Initialize authenticator
    auth = KeystrokeAuthenticator()

    print(f"⚙️  Training Configuration:")
    print(f"   Model: LSTM (2 layers, 64 hidden units)")
    print(f"   Embedding size: 32")
    print(f"   Loss function: Triplet Margin Loss (margin=1.0)")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Device: {auth.device}")
    print(f"   Training samples: {len(training_pairs)}")
    print(f"   Epochs: {epochs}")
    print()

    # Create triplet batches
    triplet_data = create_triplet_batches(training_pairs)

    if len(triplet_data) == 0:
        print("❌ ERROR: Not enough data to create triplets!")
        print("   Need at least 2 users OR multiple sessions per user.")
        sys.exit(1)

    # Train the model
    print("🏋️  Training in progress...")
    print("-" * 70)
    print()

    auth.train_model(triplet_data, epochs=epochs)

    print()
    print("-" * 70)
    print("✅ TRAINING COMPLETE!")
    print()

    return auth


def save_trained_model(auth):
    """Save the trained model and templates"""
    model_path = Path(__file__).parent / 'models' / 'trained_keystroke_model.pth'
    template_path = Path(__file__).parent / 'data' / 'trained_templates.pkl'

    print("💾 Saving trained model...")
    auth.save_model(str(model_path), str(template_path))

    print()
    print(f"   ✅ Model saved to: {model_path}")
    print(f"   ✅ Templates saved to: {template_path}")
    print()


def show_evaluation_guide():
    """Show how to evaluate the trained model"""
    print()
    print("=" * 70)
    print("📈 NEXT STEPS: EVALUATE YOUR MODEL")
    print("=" * 70)
    print()
    print("Your model is now trained! Here's how to test it:")
    print()
    print("1️⃣  Switch back to normal API server:")
    print("   - Stop api_training.py (Ctrl+C)")
    print("   - Start regular server: python backend/api.py")
    print()
    print("2️⃣  Load the trained model:")
    print("   - The API will automatically load from models/trained_keystroke_model.pth")
    print("   - If not, modify api.py to load your trained model")
    print()
    print("3️⃣  Test recognition:")
    print("   - Go to http://localhost:3000/recognize")
    print("   - Type naturally (100+ keystrokes)")
    print("   - Click 'Recognize User'")
    print()
    print("4️⃣  Expected results:")
    print("   - Your typing → 80-95% confidence (HIGH)")
    print("   - Other enrolled users → lower confidence")
    print("   - Unknown typing pattern → <60% confidence (LOW)")
    print()
    print("5️⃣  If accuracy is low:")
    print("   - Collect more training data (10+ sessions recommended)")
    print("   - Train for more epochs (50-100)")
    print("   - Ensure typing style is consistent")
    print()
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description='Train keystroke authentication model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--min-sessions', type=int, default=3,
                        help='Minimum sessions required per user (default: 3)')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("🔐 KEYSTROKE AUTHENTICATION MODEL TRAINING")
    print("=" * 70)

    # Step 1: Load training data
    training_data = load_training_sequences()

    # Step 2: Prepare training pairs
    training_pairs, valid_users = prepare_training_pairs(
        training_data,
        min_sessions=args.min_sessions
    )

    if len(training_pairs) == 0:
        print()
        print("❌ No valid training data found!")
        print("   Please collect more enrollment sessions.")
        sys.exit(1)

    # Step 3: Train the model
    auth = train_model(
        training_pairs,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Step 4: Save the trained model
    save_trained_model(auth)

    # Step 5: Show evaluation guide
    show_evaluation_guide()

    print("🎉 Training complete! Your model is ready to use.")
    print()


if __name__ == '__main__':
    main()
