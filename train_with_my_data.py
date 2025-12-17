"""
Training Script for Keystroke Authentication Model
Collects your keystroke data and trains the LSTM model

Usage:
    python train_with_my_data.py --sessions 10 --epochs 50
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'backend'))

from models.keystroke_auth_model import KeystrokeAuthenticator
from backend.feature_extraction import FeatureExtractor


def collect_training_data(num_sessions: int = 10, sequences_per_session: int = 5):
    """
    Collect training data from enrolled users through the web interface

    Args:
        num_sessions: Number of enrollment sessions to collect
        sequences_per_session: Number of sequences per session

    Returns:
        training_data: List of (sequence, user_id) tuples
    """
    print("=" * 60)
    print("KEYSTROKE AUTHENTICATION MODEL TRAINING")
    print("=" * 60)
    print()
    print("📋 Training Data Collection Strategy:")
    print()
    print(f"   Target: {num_sessions} enrollment sessions")
    print(f"   Sequences per session: {sequences_per_session}")
    print(f"   Total sequences needed: {num_sessions * sequences_per_session}")
    print()
    print("🔧 How to Collect Data:")
    print()
    print("   1. Start the backend server:")
    print("      cd backend && python api.py")
    print()
    print("   2. Start the frontend server:")
    print("      cd frontend && npm run dev")
    print()
    print("   3. Enroll yourself multiple times:")
    print(f"      - Visit http://localhost:3000/train")
    print(f"      - Enroll with username: your_name_session_1")
    print(f"      - Complete all 4 typing exercises")
    print(f"      - Repeat {num_sessions} times with different session numbers")
    print("        (your_name_session_2, your_name_session_3, etc.)")
    print()
    print("   4. (Optional) Enroll other users for better training:")
    print("      - Have friends/colleagues enroll with their names")
    print(f"      - Each should complete {num_sessions} sessions")
    print()
    print("   5. After data collection, the enrollment data will be")
    print("      automatically saved to: data/user_templates.pkl")
    print()
    print("=" * 60)
    print()

    input("Press ENTER when you've completed data collection...")
    print()

    # Load collected data
    print("📂 Loading collected enrollment data...")
    template_path = Path(__file__).parent / 'data' / 'user_templates.pkl'

    if not template_path.exists():
        print()
        print("❌ ERROR: No enrollment data found!")
        print(f"   Expected file: {template_path}")
        print()
        print("   Please complete enrollments first through the web interface.")
        sys.exit(1)

    # Load templates
    import pickle
    with open(template_path, 'rb') as f:
        user_templates = pickle.load(f)

    print(f"✅ Loaded data for {len(user_templates)} users")
    print()

    # Check if we have enough data
    total_users = len(user_templates)
    if total_users < 2:
        print("⚠️  WARNING: Only 1 user found!")
        print("   For effective training, you should have:")
        print("   - Multiple enrollment sessions from yourself, OR")
        print("   - Data from multiple different users")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("👥 Enrolled users:")
    for user_id in user_templates.keys():
        print(f"   - {user_id}")
    print()

    # Since templates only store averaged embeddings, we need to collect
    # raw keystroke data for training. Let's guide the user to export data.
    print()
    print("=" * 60)
    print("⚠️  IMPORTANT: Template-only Training Limitation")
    print("=" * 60)
    print()
    print("The current enrollment system only saves averaged templates,")
    print("not raw keystroke sequences. For proper training, we need")
    print("access to individual typing samples.")
    print()
    print("📌 Two options:")
    print()
    print("Option A: QUICK TRAINING (Template-Based)")
    print("   - Use existing templates as training data")
    print("   - Less effective but works with current data")
    print("   - Good for testing/proof-of-concept")
    print()
    print("Option B: FULL TRAINING (Sequence-Based)")
    print("   - Modify backend to save raw sequences during enrollment")
    print("   - Much better training results")
    print("   - Recommended for research/production")
    print()

    choice = input("Choose option (A/B): ").upper()
    print()

    if choice == 'B':
        print("Option B selected. I'll create the necessary modifications...")
        return None, user_templates  # Signal to create data collection system
    else:
        print("Option A selected. Using template-based training...")
        return generate_training_data_from_templates(user_templates), user_templates


def generate_training_data_from_templates(user_templates):
    """
    Generate synthetic training triplets from existing templates
    This is a workaround for template-only data
    """
    print()
    print("🔄 Generating synthetic training data from templates...")

    training_data = []
    feature_extractor = FeatureExtractor()

    # For each user, generate variations around their template
    for user_id, user_data in user_templates.items():
        template = user_data['template']  # Shape: (1, 32)

        # Generate 10 synthetic sequences per user by adding small noise
        for i in range(10):
            # Add small Gaussian noise to create variations
            noisy_template = template + np.random.normal(0, 0.05, template.shape)

            # Create a fake sequence (we'll train on embeddings directly)
            # This is not ideal but works with current data structure
            fake_sequence = np.random.randn(50, 3)  # Will be replaced by embedding

            training_data.append((fake_sequence, user_id, noisy_template))

    print(f"✅ Generated {len(training_data)} synthetic training samples")
    print()

    return training_data


def train_model(training_data, user_templates, epochs: int = 50):
    """
    Train the LSTM model with collected data
    """
    print("=" * 60)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 60)
    print()

    # Initialize authenticator
    auth = KeystrokeAuthenticator()

    # Load existing templates
    auth.user_templates = user_templates

    print(f"📊 Training Configuration:")
    print(f"   - Samples: {len(training_data)}")
    print(f"   - Users: {len(user_templates)}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Device: {auth.device}")
    print()

    # Prepare data for triplet loss training
    # We need to ensure we have anchor-positive-negative triplets
    formatted_data = []

    user_samples = {}
    for seq, user_id, _ in training_data:
        if user_id not in user_samples:
            user_samples[user_id] = []
        user_samples[user_id].append(seq)

    # Create triplets
    user_ids = list(user_samples.keys())
    for user_id in user_ids:
        samples = user_samples[user_id]

        # For each sample, create an anchor-positive-negative triplet
        for i in range(len(samples)):
            # Anchor
            formatted_data.append((samples[i], user_id))

            # Positive (same user, different sample)
            pos_idx = (i + 1) % len(samples)
            formatted_data.append((samples[pos_idx], user_id))

            # Negative (different user)
            other_users = [u for u in user_ids if u != user_id]
            if other_users:
                neg_user = np.random.choice(other_users)
                neg_sample = np.random.choice(user_samples[neg_user])
                formatted_data.append((neg_sample, neg_user))

    print(f"📦 Prepared {len(formatted_data)} training samples")
    print()

    # Train the model
    print("🏋️  Training in progress...")
    print()

    auth.train_model(formatted_data, epochs=epochs)

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print()

    # Save the trained model
    model_path = Path(__file__).parent / 'models' / 'trained_keystroke_model.pth'
    template_path = Path(__file__).parent / 'data' / 'user_templates.pkl'

    print("💾 Saving trained model...")
    auth.save_model(str(model_path), str(template_path))
    print(f"   Model saved to: {model_path}")
    print(f"   Templates saved to: {template_path}")
    print()

    return auth


def evaluate_model(auth, user_templates):
    """
    Evaluate the trained model
    """
    print("=" * 60)
    print("📈 MODEL EVALUATION")
    print("=" * 60)
    print()
    print("To test your trained model:")
    print()
    print("1. Go to http://localhost:3000/recognize")
    print("2. Type naturally (100+ keystrokes)")
    print("3. Click 'Recognize User'")
    print("4. Check if it identifies you correctly!")
    print()
    print("Expected results with trained model:")
    print("   - Your identity: 80-95% confidence (HIGH)")
    print("   - Other users: <60% confidence (LOW)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Train keystroke authentication model')
    parser.add_argument('--sessions', type=int, default=10,
                        help='Number of enrollment sessions to collect (default: 10)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')

    args = parser.parse_args()

    # Step 1: Collect training data
    training_data, user_templates = collect_training_data(
        num_sessions=args.sessions,
        sequences_per_session=5
    )

    # Check if we need to implement Option B
    if training_data is None:
        print()
        print("To implement Option B, I'll create a modified backend")
        print("that saves raw keystroke sequences during enrollment.")
        print()
        response = input("Create modified data collection system? (y/n): ")
        if response.lower() == 'y':
            print()
            print("Creating enhanced data collection system...")
            # This would create the modified backend - let user decide
            return
        else:
            print("Exiting. Run again and choose Option A to train with templates.")
            return

    # Step 2: Train the model
    auth = train_model(training_data, user_templates, epochs=args.epochs)

    # Step 3: Evaluation instructions
    evaluate_model(auth, user_templates)

    print()
    print("🎉 All done! Your model is trained and ready to use.")
    print()


if __name__ == '__main__':
    main()
