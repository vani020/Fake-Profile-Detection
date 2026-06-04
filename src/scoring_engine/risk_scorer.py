# src/scoring_engine/risk_scorer.py
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config

class RiskScorer:
    def __init__(self):
        self.config = Config()
        self.weights = self.config.WEIGHTS
        self.thresholds = self.config.RISK_THRESHOLDS
    
    def calculate_risk_score(self, photo_score, face_match_score, ratio_score, bio_score):
        """
        Calculate final risk score
        
        Args:
            photo_score: 0-1 (0 = real, 1 = fake)
            face_match_score: 0-1 (0 = match, 1 = mismatch)
            ratio_score: 0-1 (0 = normal, 1 = suspicious)
            bio_score: 0-1 (0 = clean, 1 = suspicious)
        
        Returns:
            final_score: 0-1
            risk_level: 'Low Risk', 'Medium Risk', 'High Risk'
        """
        final_score = (
            self.weights['photo'] * photo_score +
            self.weights['face_match'] * face_match_score +
            self.weights['ratio'] * ratio_score +
            self.weights['bio'] * bio_score
        )
        
        # Clamp to 0-1 range
        final_score = np.clip(final_score, 0, 1)
        
        # Determine risk level
        if final_score <= self.thresholds['low']:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
        elif final_score <= self.thresholds['medium']:
            risk_level = "🟡 Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "🔴 High Risk"
            risk_color = "red"
        
        return final_score, risk_level, risk_color
    
    def get_breakdown(self, photo_score, face_match_score, ratio_score, bio_score):
        """Get detailed score breakdown"""
        breakdown = {
            'photo_authenticity': {
                'score': photo_score,
                'weight': self.weights['photo'],
                'contribution': photo_score * self.weights['photo']
            },
            'face_matching': {
                'score': face_match_score,
                'weight': self.weights['face_match'],
                'contribution': face_match_score * self.weights['face_match']
            },
            'followers_ratio': {
                'score': ratio_score,
                'weight': self.weights['ratio'],
                'contribution': ratio_score * self.weights['ratio']
            },
            'bio_analysis': {
                'score': bio_score,
                'weight': self.weights['bio'],
                'contribution': bio_score * self.weights['bio']
            }
        }
        
        return breakdown
    
    def interpret_score(self, final_score):
        """Give interpretation of the risk score"""
        if final_score <= 0.3:
            return "✅ Profile appears genuine. All parameters within normal range."
        elif final_score <= 0.6:
            return "⚠️ Some suspicious indicators detected. Review recommended."
        else:
            return "🚨 High probability of fake/impersonation profile. Multiple red flags detected."

# Test function
if __name__ == "__main__":
    scorer = RiskScorer()
    
    # Test with sample scores
    test_cases = [
        (0.1, 0.1, 0.1, 0.1),  # Low risk
        (0.5, 0.5, 0.5, 0.5),  # Medium risk
        (0.9, 0.9, 0.9, 0.9),  # High risk
    ]
    
    for scores in test_cases:
        final, level, _ = scorer.calculate_risk_score(*scores)
        print(f"Scores: {scores} → Final: {final:.2f} → {level}")