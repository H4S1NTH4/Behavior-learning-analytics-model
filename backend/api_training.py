"""
Enhanced API endpoints for training data collection
Adds endpoints to save raw keystroke sequences during enrollment

Run this instead of api.py when collecting training data:
    python api_training.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import pickle
import json
from datetime import datetime

from models.keystroke_auth_model import KeystrokeAuthenticator
from backend.feature_extraction import FeatureExtractor

# Initialize FastAPI app
app = FastAPI(title="Keystroke Authentication API - Training Mode")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
authenticator = KeystrokeAuthenticator()
feature_extractor = FeatureExtractor()

# Training data storage
TRAINING_DATA_PATH = Path(__file__).parent.parent / 'data' / 'training_sequences.json'
TRAINING_DATA_PATH.parent.mkdir(exist_ok=True)


class EnrollmentRequest(BaseModel):
    userId: str
    keystrokeEvents: List[Dict]


class VerificationRequest(BaseModel):
    userId: str
    keystrokeEvents: List[Dict]


class IdentificationRequest(BaseModel):
    keystrokeEvents: List[Dict]
    topK: Optional[int] = 3


def load_training_data():
    """Load existing training data"""
    if TRAINING_DATA_PATH.exists():
        with open(TRAINING_DATA_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_training_sequence(user_id: str, keystroke_events: List[Dict], sequence: list):
    """Save raw keystroke sequence for training"""
    training_data = load_training_data()

    if user_id not in training_data:
        training_data[user_id] = {
            'sessions': [],
            'total_sequences': 0
        }

    # Add new session
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'keystroke_events': keystroke_events,
        'sequence': sequence.tolist(),  # Convert numpy to list for JSON
        'num_keystrokes': len(keystroke_events)
    }

    training_data[user_id]['sessions'].append(session_data)
    training_data[user_id]['total_sequences'] += 1

    # Save to file
    with open(TRAINING_DATA_PATH, 'w') as f:
        json.dump(training_data, f, indent=2)

    return training_data[user_id]['total_sequences']


@app.post("/api/auth/enroll")
async def enroll_user(request: EnrollmentRequest):
    """
    Enroll a new user with their keystroke patterns
    In training mode, this also saves raw sequences for model training
    """
    try:
        user_id = request.userId
        events = request.keystrokeEvents

        # Validate input
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="User ID is required")

        if len(events) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient training data. Need at least 200 keystrokes. Got: {len(events)}"
            )

        # Create sequence from events
        sequence = feature_extractor.create_sequence(events, sequence_length=50)

        # ⭐ TRAINING MODE: Save raw sequence
        total_sequences = save_training_sequence(user_id, events, sequence)

        # Also perform normal enrollment
        result = authenticator.enroll_user(user_id, [sequence])

        # Add training info to response
        result['training_mode'] = True
        result['total_sequences_saved'] = total_sequences

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/verify")
async def verify_user(request: VerificationRequest):
    """Verify a user's identity"""
    try:
        user_id = request.userId
        events = request.keystrokeEvents

        if not authenticator.user_templates:
            raise HTTPException(status_code=404, detail="No users enrolled yet")

        if user_id not in authenticator.user_templates:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        if len(events) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least 100 keystrokes. Got: {len(events)}"
            )

        sequence = feature_extractor.create_sequence(events, sequence_length=50)
        result = authenticator.verify_user(user_id, sequence)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/identify")
async def identify_user(request: IdentificationRequest):
    """Identify a user from keystroke pattern"""
    try:
        events = request.keystrokeEvents
        top_k = request.topK

        if not authenticator.user_templates:
            raise HTTPException(
                status_code=404,
                detail="No users enrolled yet. Please enroll at least one user first."
            )

        if len(events) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least 100 keystrokes. Got: {len(events)}"
            )

        sequence = feature_extractor.create_sequence(events, sequence_length=50)
        result = authenticator.identify_user(sequence, top_k)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/enrolled")
async def get_enrolled_users():
    """Get list of enrolled users"""
    users = list(authenticator.user_templates.keys())
    return {"users": users, "count": len(users)}


@app.get("/api/training/stats")
async def get_training_stats():
    """Get training data statistics"""
    training_data = load_training_data()

    stats = {
        'total_users': len(training_data),
        'users': {}
    }

    for user_id, user_data in training_data.items():
        stats['users'][user_id] = {
            'sessions': len(user_data['sessions']),
            'total_sequences': user_data['total_sequences'],
            'total_keystrokes': sum(s['num_keystrokes'] for s in user_data['sessions'])
        }

    return stats


@app.delete("/api/training/clear")
async def clear_training_data():
    """Clear all training data (use with caution!)"""
    if TRAINING_DATA_PATH.exists():
        TRAINING_DATA_PATH.unlink()
    return {"message": "Training data cleared"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "TRAINING",
        "enrolled_users": len(authenticator.user_templates),
        "training_data_path": str(TRAINING_DATA_PATH)
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🔧 KEYSTROKE AUTHENTICATION - TRAINING MODE")
    print("=" * 70)
    print()
    print("📊 This server saves raw keystroke sequences for model training")
    print()
    print(f"   Training data will be saved to: {TRAINING_DATA_PATH}")
    print()
    print("🌐 API endpoints:")
    print("   - POST /api/auth/enroll       (saves training data)")
    print("   - POST /api/auth/verify")
    print("   - POST /api/auth/identify")
    print("   - GET  /api/users/enrolled")
    print("   - GET  /api/training/stats    (view collected data)")
    print("   - DELETE /api/training/clear  (reset training data)")
    print()
    print("🚀 Server starting on http://localhost:8001")
    print("=" * 70)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8001)
