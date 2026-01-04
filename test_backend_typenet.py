"""
Integration test for TypeNet + Backend
Tests the complete flow: feature extraction -> TypeNet inference -> API endpoints
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from backend.feature_extraction import KeystrokeFeatureExtractor
from models.typenet_inference import TypeNetAuthenticator


def generate_sample_keystroke_events(num_events=150):
    """Generate sample keystroke events for testing"""
    events = []
    timestamp = 1000

    keycodes = [65, 83, 68, 70, 71, 72, 74, 75, 76]  # A-L keys

    for i in range(num_events):
        event = {
            'userId': 'test_user',
            'sessionId': 'session_001',
            'timestamp': timestamp,
            'key': chr(keycodes[i % len(keycodes)]),
            'dwellTime': int(np.random.uniform(50, 150)),
            'flightTime': int(np.random.uniform(20, 100)),
            'keyCode': keycodes[i % len(keycodes)]
        }
        events.append(event)
        timestamp += event['dwellTime'] + event['flightTime']

    return events


def test_feature_extraction():
    """Test 1: Feature extraction for TypeNet"""
    print("\n" + "="*60)
    print("TEST 1: Feature Extraction for TypeNet")
    print("="*60)

    extractor = KeystrokeFeatureExtractor()
    sample_events = generate_sample_keystroke_events(70)

    # Test TypeNet sequence creation
    sequence = extractor.create_typenet_sequence(sample_events, sequence_length=70)

    print(f"✅ Input events: {len(sample_events)}")
    print(f"✅ Output sequence shape: {sequence.shape}")
    print(f"✅ Expected shape: (70, 5)")

    if sequence.shape == (70, 5):
        print("✅ Shape matches TypeNet requirements!")
    else:
        print(f"❌ Shape mismatch! Got {sequence.shape}, expected (70, 5)")
        return False

    # Verify features
    print(f"\n📊 Sample features (first keystroke):")
    print(f"   HL (Hold Latency): {sequence[0, 0]:.4f}")
    print(f"   IL (Inter-key Latency): {sequence[0, 1]:.4f}")
    print(f"   PL (Press Latency): {sequence[0, 2]:.4f}")
    print(f"   RL (Release Latency): {sequence[0, 3]:.4f}")
    print(f"   KeyCode: {sequence[0, 4]:.4f}")

    return True


def test_typenet_inference():
    """Test 2: TypeNet model inference"""
    print("\n" + "="*60)
    print("TEST 2: TypeNet Inference")
    print("="*60)

    # Check if model exists
    model_path = 'models/typenet_pretrained.pth'

    if not os.path.exists(model_path):
        print(f"⚠️  TypeNet model not found at: {model_path}")
        print("   This is expected if you haven't trained the model yet.")
        print("   The model will be initialized with random weights.")
        model_path = None
    else:
        print(f"✅ Found TypeNet model at: {model_path}")

    # Initialize authenticator
    try:
        auth = TypeNetAuthenticator(model_path=model_path, device='cpu')
        print("✅ TypeNet authenticator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize authenticator: {e}")
        return False

    # Test embedding generation
    extractor = KeystrokeFeatureExtractor()
    sample_events = generate_sample_keystroke_events(70)
    sequence = extractor.create_typenet_sequence(sample_events, sequence_length=70)

    try:
        embedding = auth.get_embedding(sequence)
        print(f"✅ Generated embedding shape: {embedding.shape}")
        print(f"✅ Expected shape: (128,)")

        if embedding.shape == (128,):
            print("✅ Embedding generation successful!")
        else:
            print(f"❌ Embedding shape mismatch! Got {embedding.shape}, expected (128,)")
            return False

        print(f"\n📊 Embedding sample (first 10 values):")
        print(f"   {embedding[:10]}")

    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return False

    return True


def test_enrollment_verification():
    """Test 3: User enrollment and verification"""
    print("\n" + "="*60)
    print("TEST 3: Enrollment and Verification")
    print("="*60)

    # Initialize
    model_path = 'models/typenet_pretrained.pth' if os.path.exists('models/typenet_pretrained.pth') else None
    auth = TypeNetAuthenticator(model_path=model_path, device='cpu')
    extractor = KeystrokeFeatureExtractor()

    # Generate enrollment data (5 sequences for user)
    print("\n📝 Enrolling test users...")
    users = ['alice', 'bob', 'charlie']

    for user_id in users:
        enrollment_sequences = []
        for _ in range(5):
            events = generate_sample_keystroke_events(70)
            sequence = extractor.create_typenet_sequence(events, sequence_length=70)
            enrollment_sequences.append(sequence)

        result = auth.enroll_user(user_id, enrollment_sequences)

        if result['success']:
            print(f"   ✅ {user_id}: {result['message']}")
        else:
            print(f"   ❌ {user_id}: {result['message']}")
            return False

    # Test verification
    print("\n🔐 Testing verification...")
    for user_id in users:
        # Generate test sequence
        test_events = generate_sample_keystroke_events(70)
        test_sequence = extractor.create_typenet_sequence(test_events, sequence_length=70)

        result = auth.verify_user(user_id, test_sequence, threshold=0.5)

        status = "✅" if result['authenticated'] else "⚠️"
        print(f"   {status} {user_id}: Similarity={result['similarity']:.3f}, Risk={result['risk_score']:.3f}")

    # Test identification
    print("\n🔍 Testing identification...")
    unknown_events = generate_sample_keystroke_events(70)
    unknown_sequence = extractor.create_typenet_sequence(unknown_events, sequence_length=70)

    result = auth.identify_user(unknown_sequence, top_k=3)

    if result['success']:
        print(f"   Confidence Level: {result['confidence_level']}")
        print(f"   Top matches:")
        for match in result['matches']:
            print(f"      {match['rank']}. {match['userId']}: {match['confidence']:.1f}%")
    else:
        print(f"   ❌ {result['message']}")
        return False

    # Test template saving/loading
    print("\n💾 Testing template persistence...")
    try:
        template_path = 'models/test_user_templates.pkl'
        auth.save_templates(template_path)
        print(f"   ✅ Templates saved to {template_path}")

        # Load in new instance
        auth_new = TypeNetAuthenticator(model_path=model_path, device='cpu')
        auth_new.load_templates(template_path)
        print(f"   ✅ Templates loaded: {len(auth_new.user_templates)} users")

        # Clean up
        if os.path.exists(template_path):
            os.remove(template_path)
            print(f"   ✅ Test template file cleaned up")

    except Exception as e:
        print(f"   ❌ Template save/load failed: {e}")
        return False

    return True


def test_api_format_compatibility():
    """Test 4: Verify compatibility with API request format"""
    print("\n" + "="*60)
    print("TEST 4: API Format Compatibility")
    print("="*60)

    extractor = KeystrokeFeatureExtractor()

    # Simulate API request data format
    api_events = generate_sample_keystroke_events(150)

    print(f"✅ Generated {len(api_events)} events (API format)")

    # Test enrollment format (multiple sequences)
    print("\n📝 Testing enrollment format...")
    sequence_length = 70
    sequences = []

    for i in range(0, len(api_events) - sequence_length, sequence_length // 2):
        sequence_events = api_events[i:i + sequence_length]
        sequence = extractor.create_typenet_sequence(sequence_events, sequence_length)
        sequences.append(sequence)

    print(f"   ✅ Created {len(sequences)} sequences from {len(api_events)} events")
    print(f"   ✅ Each sequence shape: {sequences[0].shape}")

    if len(sequences) >= 3:
        print(f"   ✅ Sufficient sequences for enrollment (minimum 3)")
    else:
        print(f"   ⚠️  Only {len(sequences)} sequences, need at least 3")

    # Test verification format (single sequence)
    print("\n🔐 Testing verification format...")
    verify_events = api_events[:70]
    verify_sequence = extractor.create_typenet_sequence(verify_events, sequence_length=70)
    print(f"   ✅ Verification sequence shape: {verify_sequence.shape}")

    return True


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("TypeNet + Backend Integration Test Suite")
    print("="*60)

    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("TypeNet Inference", test_typenet_inference),
        ("Enrollment & Verification", test_enrollment_verification),
        ("API Format Compatibility", test_api_format_compatibility)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! TypeNet integration is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Train TypeNet model in Google Colab")
        print("   2. Download typenet_pretrained.pth to models/ folder")
        print("   3. Start the backend: python backend/api.py")
        print("   4. Test with real keystroke data from frontend")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
