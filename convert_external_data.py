"""
Convert External Keystroke Datasets to Training Format

This script converts external keystroke dynamics datasets to the format
expected by train_model.py (training_sequences.json).

Supported datasets:
1. DSL-StrongPasswordData.csv - Fixed password keystroke dynamics
2. keystrokes_dataset/ - Free-form text keystroke dynamics

Usage:
    # Convert DSL dataset
    python convert_external_data.py --dataset dsl --output data/training_sequences.json

    # Convert keystrokes dataset
    python convert_external_data.py --dataset keystrokes --output data/training_sequences.json

    # Convert both (combined)
    python convert_external_data.py --dataset both --output data/training_sequences.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys


def reconstruct_sequence_from_dsl(row):
    """
    Reconstruct keystroke sequence from DSL dataset timing features

    DSL format: H.x (hold time), DD.x.y (down-down time), UD.x.y (up-down time)
    Password: .tie5Roanl

    Returns: [[key_code, press_time, release_time], ...]
    """
    password = ".tie5Roanl"
    keys = list(password)

    sequence = []
    current_time = 0.0

    for i, char in enumerate(keys):
        key_code = ord(char)

        # Get hold time for this key
        col_prefix = f"H.{char}" if i > 0 else "H.period"
        hold_time = row.get(col_prefix, 0.1) * 1000  # Convert to ms

        press_time = current_time
        release_time = press_time + hold_time

        sequence.append([key_code, press_time, release_time])

        # Get down-down time to next key
        if i < len(keys) - 1:
            next_char = keys[i + 1]
            if i == 0:
                dd_col = f"DD.period.{next_char}"
            else:
                dd_col = f"DD.{char}.{next_char}"

            dd_time = row.get(dd_col, 0.2) * 1000  # Convert to ms
            current_time += dd_time

    return sequence


def convert_dsl_dataset(dsl_path, min_sessions=3, max_users=None):
    """
    Convert DSL-StrongPasswordData.csv to training format

    Args:
        dsl_path: Path to DSL CSV file
        min_sessions: Minimum sessions per user to include
        max_users: Maximum number of users to include (None = all)

    Returns:
        dict: Training data in expected format
    """
    print("📂 Loading DSL-StrongPasswordData.csv...")
    df = pd.read_csv(dsl_path)

    print(f"   Total samples: {len(df)}")
    print(f"   Total subjects: {df['subject'].nunique()}")

    training_data = {}

    # Group by subject
    for subject_id, group in df.groupby('subject'):
        sessions = []

        for idx, row in group.iterrows():
            # Reconstruct keystroke sequence
            sequence = reconstruct_sequence_from_dsl(row)

            session_data = {
                'timestamp': datetime.now().isoformat(),
                'keystroke_events': [],  # DSL doesn't have raw events
                'sequence': sequence,
                'num_keystrokes': len(sequence),
                'source': 'DSL-StrongPassword',
                'session_index': int(row['sessionIndex']),
                'rep': int(row['rep'])
            }

            sessions.append(session_data)

        # Only include users with enough sessions
        if len(sessions) >= min_sessions:
            training_data[subject_id] = {
                'sessions': sessions,
                'total_sequences': len(sessions)
            }
            print(f"   ✅ {subject_id}: {len(sessions)} sessions")
        else:
            print(f"   ⚠️  {subject_id}: {len(sessions)} sessions (skipped, need {min_sessions}+)")

        # Limit number of users if specified
        if max_users and len(training_data) >= max_users:
            print(f"   ℹ️  Reached max users limit ({max_users})")
            break

    print()
    print(f"✅ Converted {len(training_data)} users from DSL dataset")

    return training_data


def reconstruct_sequence_from_word(word, word_hold, flight_times):
    """
    Reconstruct keystroke sequence from word-level data

    Args:
        word: The typed word
        word_hold: Total hold time for the word (ms)
        flight_times: List of flight times between keys (ms)

    Returns:
        [[key_code, press_time, release_time], ...]
    """
    chars = list(word)
    n = len(chars)

    if n == 0:
        return []

    # Estimate hold time per character (uniform distribution)
    hold_per_char = word_hold / n if n > 0 else 50.0

    sequence = []
    current_time = 0.0

    for i, char in enumerate(chars):
        key_code = ord(char)

        press_time = current_time
        release_time = press_time + hold_per_char

        sequence.append([key_code, press_time, release_time])

        # Add flight time to next key
        if i < len(flight_times) and i < n - 1:
            flight = flight_times[i] if not pd.isna(flight_times[i]) else 100.0
            current_time += flight
        else:
            current_time = release_time

    return sequence


def convert_keystrokes_dataset(keystrokes_dir, min_sessions=3, max_users=None):
    """
    Convert keystrokes_dataset/ to training format

    Args:
        keystrokes_dir: Path to keystrokes_dataset directory
        min_sessions: Minimum sessions per user to include
        max_users: Maximum number of users to include (None = all)

    Returns:
        dict: Training data in expected format
    """
    print("📂 Loading keystrokes_dataset/...")
    keystrokes_path = Path(keystrokes_dir)

    if not keystrokes_path.exists():
        print(f"   ❌ Directory not found: {keystrokes_path}")
        return {}

    training_data = {}

    # Iterate through subject directories
    subject_dirs = sorted([d for d in keystrokes_path.iterdir() if d.is_dir()])
    print(f"   Found {len(subject_dirs)} subjects")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        csv_files = list(subject_dir.glob("*.csv"))

        print(f"   Processing {subject_id}: {len(csv_files)} files...")

        sessions = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Combine all words into one sequence
                full_sequence = []

                for _, row in df.iterrows():
                    word = str(row['word'])
                    word_hold = row['word_hold']

                    # Get flight times
                    flight_cols = [f'avg_flight{i}' for i in range(1, 5)]
                    flight_times = [row.get(col, 100.0) for col in flight_cols]

                    # Reconstruct sequence for this word
                    word_seq = reconstruct_sequence_from_word(word, word_hold, flight_times)

                    # Adjust timing to continue from previous word
                    if full_sequence:
                        last_time = full_sequence[-1][2]  # Last release time
                        offset = last_time + 200.0  # 200ms gap between words
                        word_seq = [[kc, pt + offset, rt + offset] for kc, pt, rt in word_seq]

                    full_sequence.extend(word_seq)

                # Only include sessions with enough keystrokes
                if len(full_sequence) >= 50:
                    session_data = {
                        'timestamp': datetime.now().isoformat(),
                        'keystroke_events': [],
                        'sequence': full_sequence,
                        'num_keystrokes': len(full_sequence),
                        'source': 'keystrokes_dataset',
                        'file': csv_file.name
                    }
                    sessions.append(session_data)

            except Exception as e:
                print(f"      ⚠️  Error processing {csv_file.name}: {e}")
                continue

        # Only include users with enough sessions
        if len(sessions) >= min_sessions:
            training_data[subject_id] = {
                'sessions': sessions,
                'total_sequences': len(sessions)
            }
            print(f"   ✅ {subject_id}: {len(sessions)} sessions")
        else:
            print(f"   ⚠️  {subject_id}: {len(sessions)} sessions (skipped, need {min_sessions}+)")

        # Limit number of users if specified
        if max_users and len(training_data) >= max_users:
            print(f"   ℹ️  Reached max users limit ({max_users})")
            break

    print()
    print(f"✅ Converted {len(training_data)} users from keystrokes_dataset")

    return training_data


def main():
    parser = argparse.ArgumentParser(description='Convert external datasets to training format')
    parser.add_argument('--dataset', choices=['dsl', 'keystrokes', 'both'], default='both',
                        help='Which dataset to convert')
    parser.add_argument('--output', default='data/training_sequences.json',
                        help='Output path for training_sequences.json')
    parser.add_argument('--min-sessions', type=int, default=3,
                        help='Minimum sessions per user (default: 3)')
    parser.add_argument('--max-users', type=int, default=None,
                        help='Maximum users to include (default: all)')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("🔄 EXTERNAL DATASET CONVERTER")
    print("=" * 70)
    print()

    training_data = {}

    # Convert DSL dataset
    if args.dataset in ['dsl', 'both']:
        dsl_path = Path('externaldata/DSL-StrongPasswordData.csv')
        if dsl_path.exists():
            dsl_data = convert_dsl_dataset(dsl_path, args.min_sessions, args.max_users)
            training_data.update(dsl_data)
        else:
            print(f"⚠️  DSL dataset not found at: {dsl_path}")
            print()

    # Convert keystrokes dataset
    if args.dataset in ['keystrokes', 'both']:
        keystrokes_dir = Path('externaldata/keystrokes_dataset')
        if keystrokes_dir.exists():
            keystrokes_data = convert_keystrokes_dataset(keystrokes_dir, args.min_sessions, args.max_users)
            training_data.update(keystrokes_data)
        else:
            print(f"⚠️  Keystrokes dataset not found at: {keystrokes_dir}")
            print()

    # Save to file
    if training_data:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print()
        print("=" * 70)
        print("✅ CONVERSION COMPLETE")
        print("=" * 70)
        print()
        print(f"📊 Summary:")
        print(f"   Total users: {len(training_data)}")

        total_sessions = sum(user['total_sequences'] for user in training_data.values())
        total_keystrokes = sum(
            sum(s['num_keystrokes'] for s in user['sessions'])
            for user in training_data.values()
        )

        print(f"   Total sessions: {total_sessions}")
        print(f"   Total keystrokes: {total_keystrokes:,}")
        print()
        print(f"💾 Saved to: {output_path}")
        print()
        print("🚀 Next steps:")
        print(f"   python train_model.py --epochs 50 --batch-size 32")
        print()
    else:
        print()
        print("❌ No data converted. Check dataset paths and try again.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
