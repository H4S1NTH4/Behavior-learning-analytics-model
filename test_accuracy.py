"""
Test model accuracy with enrolled users
Shows actual similarity scores to diagnose the accuracy issue
"""
import numpy as np
from models.keystroke_auth_model import KeystrokeAuthenticator

# Load your trained model with enrolled users
print("=" * 60)
print("TESTING MODEL ACCURACY")
print("=" * 60)
print()

auth = KeystrokeAuthenticator()
auth.load_model('models/trained_keystroke_model.pth', 'data/trained_templates.pkl')

print(f"Enrolled users: {list(auth.user_templates.keys())}")
print()

# Simulate typing test for each user
print("Simulating recognition tests...")
print()

for user_id in auth.user_templates.keys():
    print(f"Testing: {user_id}")

    # Get user's template
    user_template = auth.user_templates[user_id]

    # Create a test sequence (simulating the user typing again)
    # In reality, this would be actual typing data
    test_sequence = np.random.randn(50, 3)

    # Test identification
    result = auth.identify_user(test_sequence, top_k=3)

    if result['success'] and result['best_match']:
        best = result['best_match']
        print(f"  Best match: {best['userId']}")
        print(f"  Similarity: {best['similarity']:.3f}")
        print(f"  Confidence: {best['confidence']:.1f}%")
        print(f"  Confidence Level: {result['confidence_level']}")

        # Check if correctly identified
        if best['userId'] == user_id:
            print(f"  ✅ CORRECT")
        else:
            print(f"  ❌ WRONG (should be {user_id})")
    print()

print("=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print()
print("If similarities are LOW (<0.5):")
print("  → Model trained on different people (external dataset)")
print("  → Need to RETRAIN with YOUR users' data")
print()
print("If similarities are MEDIUM (0.5-0.7):")
print("  → Model partially works")
print("  → Collect MORE enrollment samples per user")
print()
print("If similarities are HIGH (>0.8):")
print("  → Model works well!")
print("  → Users might just need more enrollment practice")
print()
