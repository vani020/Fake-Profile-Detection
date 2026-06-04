"""
Test both modules together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.module1_photo_authenticity import PhotoAuthenticityDetector, detect_fake_photo
from src.module2_face_matching import FaceMatcher, match_faces
import numpy as np
from PIL import Image

def test_photo_module():
    """Test photo detection"""
    print("\n📸 TESTING PHOTO MODULE")
    print("="*40)
    
    detector = PhotoAuthenticityDetector()
    
    # Create test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_img)
    
    # Test
    score = detector.predict(test_pil)
    result = detect_fake_photo(test_pil)
    
    print(f"Score: {score:.3f}")
    print(f"Risk: {result['risk']}")
    print(f"Message: {result['message']}")
    print("✅ Photo module OK")

def test_face_module():
    """Test face matching"""
    print("\n👤 TESTING FACE MODULE")
    print("="*40)
    
    matcher = FaceMatcher()
    
    # Create test images
    profile = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    post1 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    post2 = np.random.randint(0, 255, (350, 350, 3), dtype=np.uint8)
    
    profile_pil = Image.fromarray(profile)
    post_pils = [Image.fromarray(post1), Image.fromarray(post2)]
    
    # Test
    result = matcher.verify(profile_pil, post_pils)
    
    print(f"Score: {result['score']:.3f}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print("✅ Face module OK")

def test_integration():
    """Test both together"""
    print("\n🔄 TESTING INTEGRATION")
    print("="*40)
    
    # Create test data
    profile = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    posts = [
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
        np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    ]
    
    profile_pil = Image.fromarray(profile)
    post_pils = [Image.fromarray(p) for p in posts]
    
    # Module 1
    detector = PhotoAuthenticityDetector()
    photo_score = detector.predict(profile_pil)
    
    # Module 2
    matcher = FaceMatcher()
    face_result = matcher.verify(profile_pil, post_pils)
    
    # Results
    print(f"📸 Photo Risk: {photo_score:.3f}")
    print(f"👤 Face Match: {face_result['score']:.3f}")
    print(f"📊 Final: Photo={'High' if photo_score>0.6 else 'Medium' if photo_score>0.3 else 'Low'}, Face={face_result['status']}")
    print("✅ Integration OK")

if __name__ == "__main__":
    print("🚀 TESTING FAKE PROFILE MODULES")
    print("="*50)
    
    test_photo_module()
    test_face_module()
    test_integration()
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")