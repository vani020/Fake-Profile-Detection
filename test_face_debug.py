# simple_test.py
import cv2
import sys
sys.path.append(r"C:\Rashika\fake profile detetction")

print("1. Importing modules...")
from src.module2_face_matching.face_matching import FaceMatchingModel

print("2. Creating model...")
matcher = FaceMatchingModel()

print("3. Loading image...")
img_path = r"C:\Rashika\fake profile detetction\datasets\real_vs_fake\real\real_face_1.jpg"

import os
if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    # List files in real folder
    real_folder = r"C:\Rashika\fake profile detetction\datasets\real_vs_fake\real"
    if os.path.exists(real_folder):
        files = os.listdir(real_folder)
        print(f"Files in real folder: {files[:5]}")
    exit()

img = cv2.imread(img_path)
if img is None:
    print("❌ Could not load image")
    exit()

print("4. Converting to RGB...")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("5. Extracting embedding...")
emb = matcher.extract_face_embedding(img)

if emb is None:
    print("❌ No face detected")
else:
    print(f"✅ Face detected! Embedding shape: {emb.shape}")

print("Done!")