# src/module2_face_matching/face_matching.py
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

class FaceMatchingModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.mtcnn = None
        self.resnet = None
        
        # OpenCV face detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.mtcnn = MTCNN(image_size=160, margin=40, device=self.device, keep_all=False)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.model_loaded = True
            print(f"✅ Face matching model loaded")
        except ImportError:
            print("⚠️ facenet-pytorch not installed")
    
    def detect_face_multi(self, image):
        """Multiple face detection methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
        if len(faces) == 0:
            faces = self.face_cascade_alt.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
        if len(faces) == 0:
            faces = self.face_cascade_profile.detectMultiScale(gray, 1.05, 5, minSize=(80, 80))
        
        return faces
    
    def detect_face_opencv(self, image):
        """Detect face using OpenCV with better parameters"""
        faces = self.detect_face_multi(image)
        
        if len(faces) > 0:
            if len(faces) > 1:
                areas = [w*h for (x, y, w, h) in faces]
                largest_idx = np.argmax(areas)
                x, y, w, h = faces[largest_idx]
            else:
                x, y, w, h = faces[0]
            
            margin = 50
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2*margin)
            h = min(image.shape[0] - y, h + 2*margin)
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            return face
        return None
    
    def align_face(self, face_img):
        """Align face for better matching"""
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(eyes) >= 2:
                eye_centers = []
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_centers.append((ex + ew//2, ey + eh//2))
                
                dx = eye_centers[1][0] - eye_centers[0][0]
                dy = eye_centers[1][1] - eye_centers[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                if abs(angle) > 5:
                    center = (face_img.shape[1]//2, face_img.shape[0]//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    face_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
            return face_img
        except:
            return face_img
    
    def enhance_face(self, face_img):
        """Enhance face image quality"""
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def extract_face_embedding(self, image):
        """Extract face embedding with multiple fallbacks"""
        if not self.model_loaded:
            return None
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                scale = max(160/h, 160/w)
                new_h, new_w = int(h*scale), int(w*scale)
                image = cv2.resize(image, (new_w, new_h))
            
            if image.dtype == np.uint8:
                image_norm = image.astype(np.float32) / 255.0
            else:
                image_norm = image
            
            img_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float().to(self.device)
        else:
            return None
        
        face = None
        try:
            if self.mtcnn:
                face = self.mtcnn(img_tensor)
        except:
            pass
        
        if face is None:
            face_opencv = self.detect_face_opencv(image)
            if face_opencv is not None:
                face_opencv = self.enhance_face(face_opencv)
                face_opencv = self.align_face(face_opencv)
                face_tensor = torch.from_numpy(face_opencv).permute(2, 0, 1).float().to(self.device) / 255.0
                face = face_tensor
        
        if face is None:
            face = img_tensor
        
        try:
            with torch.no_grad():
                if face.dim() == 3:
                    face = face.unsqueeze(0)
                embedding = self.resnet(face)
            return embedding.cpu().numpy().flatten()
        except:
            return None
    
    def compare_faces(self, image1, image2):
        """Compare two faces"""
        if not self.model_loaded:
            return 0.5, 0.5
        
        emb1 = self.extract_face_embedding(image1)
        emb2 = self.extract_face_embedding(image2)
        
        if emb1 is None or emb2 is None:
            return 0.5, 0.5
        
        similarity = float(cosine_similarity([emb1], [emb2])[0][0])
        mismatch_score = 1.0 - similarity
        
        return mismatch_score, similarity
    
    def match_profile_with_posts(self, dp_image, post_images):
        """Match DP with multiple post images"""
        if not post_images:
            return 0.5, "No post images", []
        
        scores = []
        similarities = []
        details = []
        
        for i, post in enumerate(post_images):
            score, sim = self.compare_faces(dp_image, post)
            scores.append(score)
            similarities.append(sim)
            
            # Status messages for each post
            if sim > 0.60:
                status = "✅ Strong Match"
            elif sim > 0.40:
                status = "✅ Good Match (angle variation)"
            elif sim > 0.30:
                status = "⚠️ Possible Match"
            else:
                status = "❌ Mismatch"
            
            details.append(f"Post {i+1}: {status} (conf: {sim:.2f})")
        
        if not scores:
            return 0.5, "No faces detected", []
        
        avg_score = float(np.mean(scores))
        best_sim = float(max(similarities))
        
        # Final status - LOWER THRESHOLDS FOR ANGLE VARIATION
        if best_sim > 0.55:
            status = "✅ SAME PERSON"
            result_msg = f"{status} | Confidence: {best_sim:.2f}"
        elif best_sim > 0.40:
            status = "✅ SAME PERSON (angle variation)"
            result_msg = f"{status} | Confidence: {best_sim:.2f}"
        elif best_sim > 0.30:
            status = "⚠️ POSSIBLE SAME PERSON"
            result_msg = f"{status} | Confidence: {best_sim:.2f}"
        else:
            status = "❌ DIFFERENT PERSON"
            result_msg = f"{status} | Confidence: {best_sim:.2f}"
        
        return avg_score, result_msg, details