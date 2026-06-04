# src/module1_photo_authencity/train_cnn.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

print("="*70)
print("🚀 FAKE PROFILE DETECTION - MODULE 1 TRAINING")
print("="*70)

# FULL PATH
MODEL_PATH = r"C:\Rashika\fake profile detetction\models\cnn_photo_model.h5"
os.makedirs(r"C:\Rashika\fake profile detetction\models", exist_ok=True)

class PhotoAuthenticityModel:
    def __init__(self):
        self.img_size = (224, 224)
        self.model = None
        self.model_path = MODEL_PATH
        
        # Dataset paths
        base_path = r"C:\Rashika\fake profile detetction\datasets"
        self.fake_path = os.path.join(base_path, "archive (1)", "thispersondoesnotexist.10k")
        self.real_path = os.path.join(base_path, "real_vs_fake", "real")
        
        # Create folders
        os.makedirs(r"C:\Rashika\fake profile detetction\models", exist_ok=True)
        os.makedirs(self.real_path, exist_ok=True)
    
    def load_images(self, folder_path, label, max_images=None):
        """Load images from folder - max_images=None means load ALL images"""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"   ❌ Folder not found: {folder_path}")
            return images, labels
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            try:
                image_files.extend([f for f in os.listdir(folder_path) if f.endswith(ext)])
            except:
                continue
        
        # If max_images is specified and less than total, limit it
        if max_images is not None and len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        print(f"   Found {len(image_files)} images", end="")
        
        loaded = 0
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
                    loaded += 1
            except:
                continue
        
        print(f" → Loaded {loaded}")
        return images, labels
    
    def load_all_data(self, max_fake_images=10000, max_real_images=None):
        """Load real and fake images - max_real_images=None means load ALL real faces"""
        print("\n📊 LOADING IMAGES...")
        print("-"*50)
        
        # Load fake faces (limit to 10000)
        print("\n🔴 Loading FAKE faces...")
        fake_images, fake_labels = self.load_images(self.fake_path, 1, max_images=max_fake_images)
        
        # Load ALL real faces (no limit)
        print("\n🟢 Loading ALL REAL faces...")
        real_images, real_labels = self.load_images(self.real_path, 0, max_images=max_real_images)
        
        # Combine
        all_images = fake_images + real_images
        all_labels = fake_labels + real_labels
        
        print("\n" + "="*70)
        print("📈 DATA SUMMARY")
        print("="*70)
        print(f"   TOTAL IMAGES: {len(all_images)}")
        print(f"   Fake faces (label 1): {len(fake_images)}")
        print(f"   Real faces (label 0): {len(real_images)}")
        print("="*70)
        
        if len(real_images) == 0:
            print("\n⚠️ WARNING: No real faces found!")
            print(f"   Please add real face images to: {self.real_path}")
        
        return np.array(all_images), np.array(all_labels)
    
    def prepare_data(self, images, labels, test_size=0.2):
        """Split data into train and validation"""
        if len(images) == 0:
            return None, None, None, None
        
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\n📊 DATA SPLIT:")
        print(f"   Training: {len(X_train)} images")
        print(f"   Validation: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self):
        """Build CNN model"""
        print("\n🏗️ BUILDING MODEL...")
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model built!")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
        """Train the model"""
        if X_train is None or len(X_train) == 0:
            print("❌ No training data!")
            return None
        
        if self.model is None:
            self.build_model()
        
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        print("\n🚀 STARTING TRAINING...")
        print("-"*50)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        # Manually save one more time
        self.model.save(self.model_path)
        print(f"✅ Model manually saved to: {self.model_path}")
        
        # Verify save
        if os.path.exists(self.model_path):
            size = os.path.getsize(self.model_path) / (1024*1024)
            print(f"✅✅✅ MODEL SAVED SUCCESSFULLY! File size: {size:.2f} MB")
        else:
            print(f"❌❌❌ MODEL NOT SAVED! Please check permissions.")
        
        return history
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            return True
        print(f"⚠️ Model not found at {self.model_path}")
        return False
    
    def predict(self, image):
        """Predict single image"""
        if self.model is None:
            if not self.load_model():
                return 0.5
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)[0][0]
        return prediction

def train_model():
    """Main training function"""
    model = PhotoAuthenticityModel()
    
    # Load data - max_fake_images=10000, max_real_images=None (load ALL real faces)
    images, labels = model.load_all_data(max_fake_images=10000, max_real_images=None)
    
    if len(images) == 0:
        print("\n❌ No images found! Training cancelled.")
        return None, None
    
    if len(labels[labels == 0]) == 0:
        print("\n❌ No real faces found! Please add real face images to:")
        print(f"   {model.real_path}")
        return None, None
    
    # Prepare data
    X_train, X_val, y_train, y_val = model.prepare_data(images, labels)
    
    if X_train is None:
        return None, None
    
    # Build and train
    model.build_model()
    history = model.train(X_train, y_train, X_val, y_val, epochs=15)
    
    return model, history

if __name__ == "__main__":
    train_model()