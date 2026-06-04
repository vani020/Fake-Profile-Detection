# main.py
import os
import sys

def main():
    print("="*60)
    print("🔍 Fake Profile Detection System")
    print("="*60)
    
    print("\n📋 Available options:")
    print("1. Check dataset status")
    print("2. Preprocess dataset")
    print("3. Train CNN model (Module 1)")
    print("4. Run frontend application")
    print("5. Run all (preprocess + train + frontend)")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.check_dataset_status()
    
    elif choice == '2':
        print("\n📊 Preprocessing dataset...")
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        images, labels = preprocessor.load_all_data()
        preprocessor.create_train_val_split(images, labels)
        print("✅ Preprocessing completed!")
    
    elif choice == '3':
        print("\n🚀 Training CNN model...")
        from src.module1_photo_authenticity.train_cnn import train_model
        train_model()
    
    elif choice == '4':
        print("\n🎨 Starting frontend application...")
        os.system("streamlit run frontend/app.py")
    
    elif choice == '5':
        print("\n🔄 Running complete pipeline...")
        
        # Preprocess
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        images, labels = preprocessor.load_all_data()
        preprocessor.create_train_val_split(images, labels)
        
        # Train
        from src.module1_photo_authenticity.train_cnn import train_model
        train_model()
        
        # Run frontend
        os.system("streamlit run frontend/app.py")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()