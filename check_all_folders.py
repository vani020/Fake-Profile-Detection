# check_all_folders.py
import os
import pandas as pd

print("="*70)
print("🔍 COMPLETE DATASET STRUCTURE CHECK")
print("="*70)

base_path = r"C:\Rashika\fake profile detetction\datasets"

# ====================================
# 1. CHECK ARCHIVE (1) - Fake Faces
# ====================================
fake1_path = os.path.join(base_path, "archive (1)")
print(f"\n📁 1. ARCHIVE (1) - Fake Faces")
print(f"   Path: {fake1_path}")
print(f"   Exists: {os.path.exists(fake1_path)}")

fake_images_path = None
fake_count = 0

if os.path.exists(fake1_path):
    # Check for inner folders
    items = os.listdir(fake1_path)
    print(f"   Contents: {items}")
    
    for item in items:
        item_path = os.path.join(fake1_path, item)
        if os.path.isdir(item_path):
            print(f"\n   📂 Folder: {item}")
            files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
            fake_count = len(files)
            print(f"      Image files: {fake_count}")
            if fake_count > 0:
                print(f"      Sample images: {files[:3]}")
                fake_images_path = item_path

# ====================================
# 2. CHECK ARCHIVE (2) - CSV + Images
# ====================================
fake2_path = os.path.join(base_path, "archive (2)")
print(f"\n📁 2. ARCHIVE (2) - CSV + Images")
print(f"   Path: {fake2_path}")
print(f"   Exists: {os.path.exists(fake2_path)}")

if os.path.exists(fake2_path):
    items = os.listdir(fake2_path)
    print(f"   Contents: {items}")
    
    # Check CSV files
    csv_files = [f for f in items if f.endswith('.csv')]
    print(f"\n   📊 CSV Files:")
    for csv_file in csv_files:
        csv_path = os.path.join(fake2_path, csv_file)
        df = pd.read_csv(csv_path)
        print(f"      {csv_file}: {len(df)} rows")
        print(f"      Columns: {list(df.columns)}")
    
    # Check real_vs_fake folder
    real_vs_fake_folder = os.path.join(fake2_path, "real_vs_fake")
    if os.path.exists(real_vs_fake_folder):
        print(f"\n   📂 real_vs_fake folder:")
        sub_items = os.listdir(real_vs_fake_folder)
        print(f"      Contents: {sub_items}")
        
        for sub in sub_items:
            sub_path = os.path.join(real_vs_fake_folder, sub)
            if os.path.isdir(sub_path):
                files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"      📁 {sub}: {len(files)} images")
                if len(files) > 0:
                    print(f"         Sample: {files[:2]}")

# ====================================
# 3. CHECK REAL_VS_FAKE (Empty)
# ====================================
real_path = os.path.join(base_path, "real_vs_fake", "real")
print(f"\n📁 3. REAL_VS_FAKE/REAL (Your real faces folder)")
print(f"   Path: {real_path}")
print(f"   Exists: {os.path.exists(real_path)}")

if os.path.exists(real_path):
    files = [f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Images: {len(files)}")
else:
    print(f"   ⚠️ Folder not found - will create automatically")

print("\n" + "="*70)
print("📊 SUMMARY:")
print(f"   Fake faces (archive 1): {fake_count} images")
print(f"   CSV data available: {len(csv_files) if 'csv_files' in dir() else 0} files")
print("="*70)