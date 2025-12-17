"""
LSTM-based Keystroke Dynamics Authentication Model
Implements Continuous Passive Authentication (CPA) using behavioral biometrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pickle


class KeystrokeLSTM(nn.Module):
    """
    LSTM network for keystroke dynamics authentication
    Input: Sequence of keystroke features (dwell_time, flight_time, key_category)
    Output: User embedding for similarity comparison
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2, embedding_size: int = 32):
        super(KeystrokeLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Fully connected layers for embedding
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            embeddings: (batch_size, embedding_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # Generate embedding
        embedding = self.fc(last_hidden)

        return embedding


class KeystrokeAuthenticator:
    """
    Manages enrollment and verification for keystroke authentication
    Uses Siamese network approach with contrastive learning
    """

    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KeystrokeLSTM().to(self.device)

        if model_path:
            self.load_model(model_path)

        self.user_templates = {}  # Store enrolled user templates

    def enroll_user(self, user_id: str, keystroke_sequences: List[np.ndarray]) -> Dict:
        """
        Enroll a user by creating a biometric template from their typing samples

        Args:
            user_id: Unique user identifier
            keystroke_sequences: List of keystroke sequences (each of shape (seq_len, 3))

        Returns:
            enrollment_result: Dict with success status and template info
        """
        self.model.eval()

        embeddings = []
        with torch.no_grad():
            for sequence in keystroke_sequences:
                # Convert to tensor
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq_len, 3)

                # Get embedding
                embedding = self.model(x)
                embeddings.append(embedding.cpu().numpy())

        # Create user template as mean of embeddings
        template = np.mean(embeddings, axis=0)
        template_std = np.std(embeddings, axis=0)

        self.user_templates[user_id] = {
            'template': template,
            'std': template_std,
            'sample_count': len(keystroke_sequences)
        }

        return {
            'success': True,
            'user_id': user_id,
            'samples_enrolled': len(keystroke_sequences),
            'message': 'User enrolled successfully'
        }

    def verify_user(self, user_id: str, keystroke_sequence: np.ndarray, threshold: float = 0.7) -> Dict:
        """
        Verify if the keystroke pattern matches the enrolled user

        Args:
            user_id: User to verify
            keystroke_sequence: Single keystroke sequence to verify
            threshold: Similarity threshold (0-1, higher = stricter)

        Returns:
            verification_result: Dict with verification status and confidence
        """
        if user_id not in self.user_templates:
            return {
                'success': False,
                'authenticated': False,
                'message': 'User not enrolled',
                'risk_score': 1.0
            }

        self.model.eval()

        # Get embedding for current sequence
        with torch.no_grad():
            x = torch.FloatTensor(keystroke_sequence).unsqueeze(0).to(self.device)
            current_embedding = self.model(x).cpu().numpy()

        # Compare with user template
        template = self.user_templates[user_id]['template']

        # Calculate cosine similarity
        similarity = self._cosine_similarity(current_embedding[0], template[0])

        # Calculate risk score (1 - similarity)
        risk_score = 1 - similarity

        # Authenticate if similarity exceeds threshold
        authenticated = similarity >= threshold

        return {
            'success': True,
            'authenticated': authenticated,
            'user_id': user_id,
            'similarity': float(similarity),
            'risk_score': float(risk_score),
            'threshold': threshold,
            'message': 'Authenticated' if authenticated else 'Authentication failed - typing pattern mismatch'
        }

    def identify_user(self, keystroke_sequence: np.ndarray, top_k: int = 3) -> Dict:
        """
        Identify user by comparing keystroke pattern against all enrolled users
        Returns top K matching users with similarity scores

        Args:
            keystroke_sequence: Single keystroke sequence to identify (shape: (seq_len, 3))
            top_k: Number of top matches to return (default: 3)

        Returns:
            identification_result: Dict containing:
                - success: Boolean indicating if identification was attempted
                - matches: List of top K matches with userId, similarity, confidence, rank
                - best_match: Top matching user
                - confidence_level: HIGH/MEDIUM/LOW based on best match similarity
                - total_enrolled_users: Number of users in database
                - message: Status message
        """
        # Check if any users are enrolled
        if not self.user_templates:
            return {
                'success': False,
                'message': 'No users enrolled yet. Please enroll at least one user first.',
                'matches': [],
                'total_enrolled_users': 0
            }

        self.model.eval()

        # Get embedding for input sequence
        with torch.no_grad():
            x = torch.FloatTensor(keystroke_sequence).unsqueeze(0).to(self.device)
            current_embedding = self.model(x).cpu().numpy()

        # Compare against all enrolled users
        similarities = []
        for user_id, user_data in self.user_templates.items():
            template = user_data['template']

            # Calculate cosine similarity
            similarity = self._cosine_similarity(current_embedding[0], template[0])

            # Convert similarity to confidence percentage
            confidence = similarity * 100.0

            similarities.append({
                'userId': user_id,
                'similarity': float(similarity),
                'confidence': float(confidence)
            })

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Take top K matches
        num_matches = min(top_k, len(similarities))
        top_matches = similarities[:num_matches]

        # Add rank to matches
        for i, match in enumerate(top_matches):
            match['rank'] = i + 1

        # Get best match
        best_match = top_matches[0] if top_matches else None

        # Determine confidence level based on best match similarity
        if best_match:
            best_similarity = best_match['similarity']
            if best_similarity >= 0.8:
                confidence_level = 'HIGH'
                message = 'Identified with HIGH confidence'
            elif best_similarity >= 0.6:
                confidence_level = 'MEDIUM'
                message = 'Identified with MEDIUM confidence'
            else:
                confidence_level = 'LOW'
                message = 'Identified with LOW confidence - results may be unreliable'
        else:
            confidence_level = 'UNKNOWN'
            message = 'No matches found'

        return {
            'success': True,
            'matches': top_matches,
            'best_match': best_match,
            'confidence_level': confidence_level,
            'total_enrolled_users': len(self.user_templates),
            'message': message
        }

    def continuous_authentication(self, user_id: str, keystroke_sequences: List[np.ndarray],
                                  threshold: float = 0.7) -> Dict:
        """
        Perform continuous authentication over multiple sequences
        Returns overall risk score and alerts

        Args:
            user_id: User to monitor
            keystroke_sequences: Multiple recent sequences
            threshold: Similarity threshold

        Returns:
            monitoring_result: Dict with risk assessment
        """
        if not keystroke_sequences:
            return {'success': False, 'message': 'No data provided'}

        results = []
        for sequence in keystroke_sequences:
            result = self.verify_user(user_id, sequence, threshold)
            if result['success']:
                results.append(result)

        if not results:
            return {'success': False, 'message': 'Verification failed'}

        # Calculate aggregate risk
        avg_risk = np.mean([r['risk_score'] for r in results])
        max_risk = np.max([r['risk_score'] for r in results])

        # Alert levels
        alert_level = 'LOW'
        if max_risk > 0.5:
            alert_level = 'MEDIUM'
        if max_risk > 0.7:
            alert_level = 'HIGH'

        return {
            'success': True,
            'user_id': user_id,
            'average_risk_score': float(avg_risk),
            'max_risk_score': float(max_risk),
            'alert_level': alert_level,
            'authenticated': alert_level != 'HIGH',
            'samples_analyzed': len(results),
            'message': f'Risk level: {alert_level}'
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def save_model(self, model_path: str, templates_path: str = None):
        """Save model and user templates"""
        torch.save(self.model.state_dict(), model_path)

        if templates_path:
            with open(templates_path, 'wb') as f:
                pickle.dump(self.user_templates, f)

    def load_model(self, model_path: str, templates_path: str = None):
        """Load model and user templates"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        if templates_path:
            with open(templates_path, 'rb') as f:
                self.user_templates = pickle.load(f)

    def train_model(self, training_data: List[Tuple[np.ndarray, str]], epochs: int = 10):
        """
        Train the model using contrastive learning
        This is a simplified training loop - in production, use more sophisticated training

        Args:
            training_data: List of (sequence, user_id) pairs
            epochs: Number of training epochs
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.TripletMarginLoss(margin=1.0)

        print(f"Training on {len(training_data)} samples for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            # Simple triplet mining (anchor, positive, negative)
            for i in range(0, len(training_data) - 2, 3):
                # Get triplet
                anchor_seq, anchor_user = training_data[i]
                pos_seq, pos_user = training_data[i + 1]
                neg_seq, neg_user = training_data[i + 2]

                # Ensure positive is same user, negative is different
                if pos_user != anchor_user:
                    continue

                # Convert to tensors
                anchor = torch.FloatTensor(anchor_seq).unsqueeze(0).to(self.device)
                positive = torch.FloatTensor(pos_seq).unsqueeze(0).to(self.device)
                negative = torch.FloatTensor(neg_seq).unsqueeze(0).to(self.device)

                # Forward pass
                anchor_emb = self.model(anchor)
                pos_emb = self.model(positive)
                neg_emb = self.model(negative)

                # Calculate loss
                loss = criterion(anchor_emb, pos_emb, neg_emb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(training_data) // 3)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


# Example usage
if __name__ == "__main__":
    # Initialize authenticator
    auth = KeystrokeAuthenticator()

    # Simulated enrollment data (in practice, collect from real user sessions)
    user_id = "student_001"
    enrollment_sequences = [
        np.random.randn(50, 3) for _ in range(5)  # 5 enrollment samples
    ]

    # Enroll user
    result = auth.enroll_user(user_id, enrollment_sequences)
    print("Enrollment:", result)

    # Verify user
    test_sequence = np.random.randn(50, 3)
    verification = auth.verify_user(user_id, test_sequence)
    print("Verification:", verification)

    # Continuous monitoring
    monitoring_sequences = [np.random.randn(50, 3) for _ in range(10)]
    monitoring = auth.continuous_authentication(user_id, monitoring_sequences)
    print("Monitoring:", monitoring)
