"""
Test script for keystroke dynamics authentication system
Generates synthetic data and tests the complete pipeline
"""

import sys
import os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.feature_extraction import KeystrokeFeatureExtractor
from models.keystroke_auth_model import KeystrokeAuthenticator


def generate_synthetic_user_data(user_id: str, num_sessions: int = 5,
                                 base_dwell_mean: float = 80,
                                 base_flight_mean: float = 150) -> list:
    """
    Generate synthetic keystroke data that mimics a user's typing pattern
    Each user has characteristic dwell/flight times with some variance
    """
    sessions = []

    for session in range(num_sessions):
        events = []
        timestamp = 1000

        # Simulate typing ~100 characters
        for i in range(100):
            # Add noise to simulate natural variation
            dwell_time = int(np.random.normal(base_dwell_mean, 15))
            flight_time = int(np.random.normal(base_flight_mean, 30))

            # Clamp to reasonable values
            dwell_time = max(30, min(200, dwell_time))
            flight_time = max(50, min(500, flight_time))

            # Random key (simplified)
            key = random.choice('abcdefghijklmnopqrstuvwxyz ')

            event = {
                'userId': user_id,
                'sessionId': f'session_{session}',
                'timestamp': timestamp,
                'key': key,
                'dwellTime': dwell_time,
                'flightTime': flight_time,
                'keyCode': ord(key)
            }

            events.append(event)
            timestamp += dwell_time + flight_time

        sessions.append(events)

    return sessions


def test_feature_extraction():
    """Test feature extraction module"""
    print("\n" + "="*60)
    print("TEST 1: Feature Extraction")
    print("="*60)

    extractor = KeystrokeFeatureExtractor()

    # Generate sample data
    sample_events = generate_synthetic_user_data("test_user", num_sessions=1)[0]

    # Test feature extraction
    features = extractor.extract_features(sample_events)
    print(f"✓ Extracted {len(features)} features from {len(sample_events)} keystrokes")
    print(f"  Feature vector sample: {features[:5]}")

    # Test sequence creation for LSTM
    sequence = extractor.create_sequence(sample_events, sequence_length=50)
    print(f"✓ Created sequence with shape: {sequence.shape}")

    return True


def test_enrollment_and_verification():
    """Test user enrollment and verification"""
    print("\n" + "="*60)
    print("TEST 2: Enrollment and Verification")
    print("="*60)

    auth = KeystrokeAuthenticator()
    extractor = KeystrokeFeatureExtractor()

    # Create two users with different typing patterns
    user1_data = generate_synthetic_user_data("alice", num_sessions=5,
                                              base_dwell_mean=80, base_flight_mean=150)
    user2_data = generate_synthetic_user_data("bob", num_sessions=5,
                                              base_dwell_mean=120, base_flight_mean=200)

    # Prepare sequences for enrollment
    user1_sequences = []
    for session_events in user1_data:
        seq = extractor.create_sequence(session_events, sequence_length=50)
        user1_sequences.append(seq)

    user2_sequences = []
    for session_events in user2_data:
        seq = extractor.create_sequence(session_events, sequence_length=50)
        user2_sequences.append(seq)

    # Enroll users
    print("\nEnrolling users...")
    result1 = auth.enroll_user("alice", user1_sequences)
    print(f"✓ Alice enrolled: {result1['message']}")

    result2 = auth.enroll_user("bob", user2_sequences)
    print(f"✓ Bob enrolled: {result2['message']}")

    # Test verification - Alice verifying as Alice (should pass)
    print("\nTesting legitimate user verification...")
    test_alice = generate_synthetic_user_data("alice", num_sessions=1,
                                              base_dwell_mean=80, base_flight_mean=150)[0]
    test_alice_seq = extractor.create_sequence(test_alice, sequence_length=50)

    verify_result = auth.verify_user("alice", test_alice_seq, threshold=0.5)
    print(f"✓ Alice verifying as Alice:")
    print(f"  - Authenticated: {verify_result['authenticated']}")
    print(f"  - Similarity: {verify_result['similarity']:.3f}")
    print(f"  - Risk Score: {verify_result['risk_score']:.3f}")

    # Test impersonation - Bob trying to verify as Alice (should fail)
    print("\nTesting impersonation detection...")
    test_bob = generate_synthetic_user_data("bob", num_sessions=1,
                                            base_dwell_mean=120, base_flight_mean=200)[0]
    test_bob_seq = extractor.create_sequence(test_bob, sequence_length=50)

    imposter_result = auth.verify_user("alice", test_bob_seq, threshold=0.5)
    print(f"✓ Bob attempting to verify as Alice:")
    print(f"  - Authenticated: {imposter_result['authenticated']}")
    print(f"  - Similarity: {imposter_result['similarity']:.3f}")
    print(f"  - Risk Score: {imposter_result['risk_score']:.3f}")

    return True


def test_continuous_authentication():
    """Test continuous authentication monitoring"""
    print("\n" + "="*60)
    print("TEST 3: Continuous Authentication")
    print("="*60)

    auth = KeystrokeAuthenticator()
    extractor = KeystrokeFeatureExtractor()

    # Enroll user
    user_data = generate_synthetic_user_data("charlie", num_sessions=5,
                                            base_dwell_mean=90, base_flight_mean=160)
    sequences = [extractor.create_sequence(session, 50) for session in user_data]
    auth.enroll_user("charlie", sequences)
    print("✓ User 'charlie' enrolled")

    # Simulate continuous monitoring with legitimate typing
    print("\nMonitoring legitimate session...")
    legitimate_sessions = generate_synthetic_user_data("charlie", num_sessions=10,
                                                       base_dwell_mean=90, base_flight_mean=160)
    legitimate_sequences = [extractor.create_sequence(s, 50) for s in legitimate_sessions]

    result = auth.continuous_authentication("charlie", legitimate_sequences[:5])
    print(f"✓ Legitimate session analysis:")
    print(f"  - Alert Level: {result['alert_level']}")
    print(f"  - Average Risk: {result['average_risk_score']:.3f}")
    print(f"  - Max Risk: {result['max_risk_score']:.3f}")

    # Simulate impersonation (typing pattern changes)
    print("\nDetecting session takeover...")
    imposter_sessions = generate_synthetic_user_data("imposter", num_sessions=5,
                                                     base_dwell_mean=150, base_flight_mean=250)
    imposter_sequences = [extractor.create_sequence(s, 50) for s in imposter_sessions]

    result = auth.continuous_authentication("charlie", imposter_sequences)
    print(f"✓ Impersonation detected:")
    print(f"  - Alert Level: {result['alert_level']}")
    print(f"  - Average Risk: {result['average_risk_score']:.3f}")
    print(f"  - Max Risk: {result['max_risk_score']:.3f}")

    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("KEYSTROKE DYNAMICS AUTHENTICATION SYSTEM - TEST SUITE")
    print("="*60)

    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Enrollment & Verification", test_enrollment_and_verification),
        ("Continuous Authentication", test_continuous_authentication)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            results.append((test_name, f"ERROR: {str(e)}"))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, status in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {status}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
