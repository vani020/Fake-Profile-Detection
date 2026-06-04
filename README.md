# 🔍 Fake Profile Detection System

An AI-powered multi-factor fake profile detection system for social media platforms like Instagram, Facebook, and Twitter/X.

---
## 📌 Overview

This system detects fake and impersonation social media profiles using **4 different modules** with a priority-based weighted scoring system.

---
### Problem Statement
Social media platforms face a rapid increase in fake and impersonation accounts used for:
- Identity theft
- Online fraud
- Spreading misinformation
- Harassment and scams
---
### Solution
A **multi-factor intelligent system** that evaluates profile authenticity based on:
- Image analysis
- Identity consistency
- Behavioral parameters
- Content analysis

---

## 🏗️ System Architecture

### Module 1: Profile Photo Authenticity (40% Weight)
- Classifies profile images as **Real** or **Suspicious**
- Uses **CNN** with Transfer Learning (MobileNet/ResNet)
- Detects AI-generated faces, stock images, and unrealistic facial artifacts

### Module 2: DP vs Post Face Matching (30% Weight)
- Verifies identity consistency between profile picture and post images
- Uses **Face Detection** (MTCNN/OpenCV) + **Facial Embeddings** + **Cosine Similarity**
- Detects impersonation, stolen photos, and multi-person profile misuse

### Module 3: Followers Behavior Analysis (20% Weight)
- Analyzes follower/following ratio and growth patterns
- Detects sudden spikes and suspicious behavioral patterns
- Rule-based threshold logic based on research papers

### Module 4: Bio Content Analysis (10% Weight)
- Checks for empty bio, suspicious keywords, promotional text, excessive links
- Keyword-based filtering with comprehensive spam dataset

---

## 📊 Weighted Risk Score Formula
Final Risk Score = (0.40 × PhotoScore) + (0.30 × FaceMatchScore) + (0.20 × RatioScore) + (0.10 × BioScore)

---
### Risk Levels
| Score Range | Risk Level |
|-------------|------------|
| 0.00 - 0.30 | 🟢 Low Risk |
| 0.31 - 0.60 | 🟡 Medium Risk |
| 0.61 - 1.00 | 🔴 High Risk |

---

## 🛠️ Tech Stack

### Programming
- **Python 3.11+**

### Deep Learning & Computer Vision
- **TensorFlow / Keras** - CNN model training
- **MobileNetV2** - Transfer learning backbone
- **FaceNet-PyTorch** - Face recognition and embeddings
- **MTCNN** - Face detection
- **OpenCV** - Image processing

### Data Processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **Scikit-learn** - Data splitting and preprocessing

### Frontend
- **Streamlit** - Interactive web interface

### Additional Libraries
- **Plotly** - Interactive gauges and charts
- **Pillow** - Image handling
- **tqdm** - Progress bars


---

## 🚀 Installation & Setup

1. Clone the Repository
git clone https://github.com/r20j/fake-profile-detection.git
cd fake-profile-detection
2. Install Dependencies
pip install -r requirements.txt
3. Download Datasets
Dataset	Use	Size	Link
Fake Faces (AI Generated)	Training fake faces	500MB	ThisPersonDoesNotExist
Real Faces	Training real faces	1.5GB	CelebA
Place datasets in datasets/ folder with structure:
datasets/
├── archive (1)/thispersondoesnotexist.10k/    # Fake faces
└── real_vs_fake/real/                         # Real faces
4. Train the Model
cd src/module1_photo_authenticity
python train_cnn.py
5. Run the Application
streamlit run frontend/app.py
---

📝 License
This project is for academic/research purposes. The CelebA dataset is available for non-commercial research purposes only.

---

Team-DeepQuest
Team Members- Rashika Jain(Team lead)
              Vani Tyagi
              Adiba Anjhum

---
