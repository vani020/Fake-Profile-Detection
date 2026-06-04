# src/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.processed_path = self.config.PROCESSED_PATH
        
    def load_images_from_folder(self, folder_path, label, max_images=None):
        """Load images from a folder"""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            return images, labels
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"📂 Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
        
        for img_file in tqdm(image_files, desc=f"Loading {label} images"):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.config.IMG_SIZE)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                continue
        
        return images, labels
    
    def load_fake_faces(self, max_images=5000):
        """Load fake faces from archive folders"""
        all_images = []
        all_labels = []
        
        for fake_folder in self.config.FAKE_FACE_FOLDERS:
            if os.path.exists(fake_folder):
                images, labels = self.load_images_from_folder(
                    fake_folder, label=1, max_images=max_images
                )
                all_images.extend(images)
                all_labels.extend(labels)
                print(f"✅ Loaded {len(images)} fake images from {fake_folder}")
        
        return all_images, all_labels
    
    def load_real_faces(self, max_images=5000):
        """Load real faces from real_vs_fake folder"""
        all_images = []
        all_labels = []
        
        real_folder = os.path.join(self.config.REAL_VS_FAKE_PATH, "real")
        if os.path.exists(real_folder):
            images, labels = self.load_images_from_folder(
                real_folder, label=0, max_images=max_images
            )
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"✅ Loaded {len(images)} real images from {real_folder}")
        
        return all_images, all_labels
    
    def load_all_data(self, max_images_per_class=5000):
        """Load all available data"""
        print("\n📊 Loading dataset...")
        print("="*50)
        
        # Load fake faces
        fake_images, fake_labels = self.load_fake_faces(max_images_per_class)
        
        # Load real faces
        real_images, real_labels = self.load_real_faces(max_images_per_class)
        
        # Combine
        all_images = real_images + fake_images
        all_labels = real_labels + fake_labels
        
        print("="*50)
        print(f"📈 Total images loaded: {len(all_images)}")
        print(f"   - Real images: {len(real_images)}")
        print(f"   - Fake images: {len(fake_images)}")
        print("="*50)
        
        return np.array(all_images), np.array(all_labels)
    
    def create_train_val_split(self, images, labels, val_size=0.2):
        """Create train/validation split"""
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=val_size, random_state=42, stratify=labels
        )
        
        # Save processed data
        np.save(os.path.join(self.processed_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_path, 'X_val.npy'), X_val)
        np.save(os.path.join(self.processed_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_path, 'y_val.npy'), y_val)
        
        print(f"\n✅ Data split completed:")
        print(f"   - Training: {len(X_train)} images")
        print(f"   - Validation: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def check_dataset_status(self):
        """Check what datasets are available"""
        print("\n🔍 Checking dataset status...")
        print("="*50)
        
        # Check fake folders
        for folder in self.config.FAKE_FACE_FOLDERS:
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✅ Fake faces found: {folder} ({count} images)")
            else:
                print(f"❌ Fake faces not found: {folder}")
        
        # Check real folder
        real_folder = os.path.join(self.config.REAL_VS_FAKE_PATH, "real")
        if os.path.exists(real_folder):
            count = len([f for f in os.listdir(real_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ Real faces found: {real_folder} ({count} images)")
        else:
            print(f"❌ Real faces not found: {real_folder}")
        
        print("="*50)

# Run to check
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.check_dataset_status()