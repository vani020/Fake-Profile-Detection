# src/config.py
import os

class Config:
    # Paths
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_PATH, "datasets")
    PROCESSED_PATH = os.path.join(DATASET_PATH, "processed")
    MODELS_PATH = os.path.join(BASE_PATH, "models")
    
    # Create directories if not exist
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Dataset folders (based on your structure)
    FAKE_FACE_FOLDERS = [
        os.path.join(DATASET_PATH, "archive (1)"),
        os.path.join(DATASET_PATH, "archive (2)"),
    ]
    REAL_VS_FAKE_PATH = os.path.join(DATASET_PATH, "real_vs_fake")
    
    # Model parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Risk weights
    WEIGHTS = {
        'photo': 0.4,
        'face_match': 0.3,
        'ratio': 0.2,
        'bio': 0.1
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6,
        'high': 1.0
    }
    
    # Face matching threshold
    FACE_MATCH_THRESHOLD = 0.6
    
    # Bio suspicious keywords
    SUSPICIOUS_KEYWORDS = [
        'click', 'link', 'free', 'money', 'earn', 'win', 'prize',
        'giveaway', 'offer', 'discount', 'promotion', 'follow back',
        'f4f', 'l4l', 's4s', 'buy', 'sell', 'cheap'
    ]