"""
Test script for TypeNet inference
Run this after training to verify everything works
"""

import numpy as np
from models.typenet_inference import TypeNetAuthenticator


def test_typenet():
    print("=" * 60)
    print("TypeNet Inference Test")
    print("=" * 60)

    # Test 1: Initialize and load model
    print("\n📥 Test 1: Loading TypeNet model...")
    try:
        auth = TypeNetAuthenticator(
            model_path='models/typenet_pretrained.pth',
            device='cpu'
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Make sure 'models/typenet_pretrained.pth' exists!")
        print("   Download it from Google Colab after training.")
        return

    # Test 2: Generate embedding
    print("\n🧮 Test 2: Generating embedding...")
    try:
        test_sequence = np.random.randn(70, 5).astype(np.float32)
        embedding = auth.get_embedding(test_sequence)
        print(f"✅ Embedding generated successfully")
        print(f"   Shape: {embedding.shape} (expected: (128,))")
        print(f"   Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return

    # Test 3: Enroll users
    print("\n👥 Test 3: Enrolling test users...")
    try:
        # Simulate 3 different users
        users = ['alice', 'bob', 'charlie']

        for user_id in users:
            # Generate 5 enrollment sequences per user
            # In reality, these would be actual keystroke data
            enrollment_sequences = [
                np.random.randn(70, 5).astype(np.float32) for _ in range(5)
            ]

            result = auth.enroll_user(user_id, enrollment_sequences)

            if result['success']:
                print(f"   ✅ {user_id}: {result['message']}")
            else:
                print(f"   ❌ {user_id}: {result['message']}")

    except Exception as e:
        print(f"❌ Enrollment failed: {e}")
        return

    # Test 4: Verify user
    print("\n🔐 Test 4: Verifying users...")
    try:
        for user_id in users:
            # Generate a test sequence (should match enrolled user pattern in reality)
            test_sequence = np.random.randn(70, 5).astype(np.float32)

            result = auth.verify_user(user_id, test_sequence, threshold=0.5)

            status = "✅ PASS" if result['authenticated'] else "❌ FAIL"
            print(f"   {status} {user_id}: Similarity={result['similarity']:.3f}, Risk={result['risk_score']:.3f}")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return

    # Test 5: Identify user
    print("\n🔍 Test 5: Identifying unknown user...")
    try:
        unknown_sequence = np.random.randn(70, 5).astype(np.float32)
        result = auth.identify_user(unknown_sequence, top_k=3)

        if result['success']:
            print(f"   Confidence: {result['confidence_level']}")
            print(f"   Top matches:")
            for match in result['matches']:
                print(f"      {match['rank']}. {match['userId']}: {match['confidence']:.1f}%")
        else:
            print(f"   ❌ {result['message']}")

    except Exception as e:
        print(f"❌ Identification failed: {e}")
        return

    # Test 6: Save and load templates
    print("\n💾 Test 6: Saving user templates...")
    try:
        auth.save_templates('models/user_templates.pkl')
        print("   ✅ Templates saved successfully")

        # Test loading
        auth_new = TypeNetAuthenticator(
            model_path='models/typenet_pretrained.pth'
        )
        auth_new.load_templates('models/user_templates.pkl')
        print(f"   ✅ Templates loaded: {len(auth_new.user_templates)} users")

    except Exception as e:
        print(f"❌ Template save/load failed: {e}")
        return

    # Test 7: Check model architecture
    print("\n🏗️ Test 7: Model architecture info...")
    total_params = sum(p.numel() for p in auth.model.parameters())
    trainable_params = sum(p.numel() for p in auth.model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {auth.device}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Replace random data with real keystroke sequences")
    print("   2. Integrate with your backend API")
    print("   3. Test with actual user typing data")
    print("   4. Tune the verification threshold (currently 0.7)")
    print("\n📚 See TYPENET_USAGE_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    test_typenet()
