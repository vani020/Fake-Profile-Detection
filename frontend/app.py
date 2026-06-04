# frontend/app.py
import streamlit as st
import cv2
import numpy as np
import sys
import os
import pandas as pd
from PIL import Image
import time
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.module1_photo_authenticity.train_cnn import PhotoAuthenticityModel
from src.module2_face_matching.face_matching import FaceMatchingModel

st.set_page_config(page_title="Fake Profile Detector", page_icon="🔍", layout="wide")

# Session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle in sidebar
with st.sidebar:
    dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()

# ========== CSS - Both Light & Dark Mode ==========
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .dashboard-title { font-size: 2.5rem; font-weight: bold; }
        .dashboard-subtitle { font-size: 1rem; opacity: 0.9; }
        .card {
            background: #1e1e2e;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
            border: 1px solid #333;
        }
        .card-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #fff;
            border-left: 4px solid #667eea;
            padding-left: 0.8rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #2a2a3a;
            border-radius: 20px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid #444;
        }
        .metric-value { font-size: 2rem; font-weight: bold; color: #667eea; }
        .metric-label { color: #aaa; margin-top: 0.5rem; }
        .risk-low {
            background: linear-gradient(135deg, #1a472a 0%, #0e2a1a 100%);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #2ecc71;
        }
        .risk-medium {
            background: linear-gradient(135deg, #4a3a1a 0%, #2a1e0a 100%);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #f39c12;
        }
        .risk-high {
            background: linear-gradient(135deg, #4a1a1a 0%, #2a0a0a 100%);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #e74c3c;
        }
        .risk-score { font-size: 2rem; font-weight: bold; color: white; }
        .risk-label { color: rgba(255,255,255,0.9); margin-top: 0.5rem; }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            padding: 0.8rem;
            font-weight: bold;
            width: 100%;
        }
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            margin: 1rem 0;
            border-radius: 3px;
        }
        .stMarkdown, p, span, label, .stTextArea label, .stNumberInput label {
            color: #fff !important;
        }
        .stTextArea textarea, .stNumberInput input {
            background: #2a2a3a !important;
            color: white !important;
            border: 1px solid #667eea !important;
        }
        [data-testid="stSidebar"] {
            background: #0f0f1a;
            border-right: 1px solid #333;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%); }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .dashboard-title { font-size: 2.5rem; font-weight: bold; }
        .dashboard-subtitle { font-size: 1rem; opacity: 0.9; }
        .card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .card-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #333;
            border-left: 4px solid #667eea;
            padding-left: 0.8rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 20px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .metric-value { font-size: 2rem; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-top: 0.5rem; }
        .risk-low {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #28a745;
        }
        .risk-medium {
            background: linear-gradient(135deg, #fff3cd, #ffeeba);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #ffc107;
        }
        .risk-high {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #dc3545;
        }
        .risk-score { font-size: 2rem; font-weight: bold; color: #333; }
        .risk-label { color: #555; margin-top: 0.5rem; }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            padding: 0.8rem;
            font-weight: bold;
            width: 100%;
        }
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            margin: 1rem 0;
            border-radius: 3px;
        }
    </style>
    """, unsafe_allow_html=True)

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    with st.spinner("🚀 Loading AI Models..."):
        photo_model = PhotoAuthenticityModel()
        photo_model.load_model()
        face_matcher = FaceMatchingModel()
    return photo_model, face_matcher

# ========== HEADER ==========
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">🔍 Fake Profile Detection System</div>
    <div class="dashboard-subtitle">AI-Powered | Multi-Factor Analysis | Real-time Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    platform = st.selectbox("🌐 Platform", ["Instagram", "Facebook", "Twitter/X", "LinkedIn", "TikTok"])
    
    st.markdown("#### 📊 Risk Weights")
    photo_weight = st.slider("📸 Photo", 0.0, 1.0, 0.40, 0.05)
    face_weight = st.slider("👤 Face", 0.0, 1.0, 0.30, 0.05)
    ratio_weight = st.slider("📊 Behavior", 0.0, 1.0, 0.20, 0.05)
    bio_weight = st.slider("📝 Bio", 0.0, 1.0, 0.10, 0.05)
    
    total = photo_weight + face_weight + ratio_weight + bio_weight
    if total > 0:
        weights = {
            'photo': photo_weight / total,
            'face': face_weight / total,
            'ratio': ratio_weight / total,
            'bio': bio_weight / total
        }
    else:
        weights = {'photo': 0.40, 'face': 0.30, 'ratio': 0.20, 'bio': 0.10}
    
    st.markdown("---")
    st.metric("📈 Total Analyses", len(st.session_state.analysis_history))
    st.markdown("---")
    st.markdown("### 📊 Risk Scale")
    st.markdown("🟢 **0.00-0.30** → Low Risk")
    st.markdown("🟡 **0.31-0.60** → Medium Risk")
    st.markdown("🔴 **0.61-1.00** → High Risk")

# ========== MAIN CONTENT ==========
st.markdown("### 📋 Profile Analysis")
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Profile Photo</div>', unsafe_allow_html=True)
    profile_photo = st.file_uploader("Upload profile picture", type=['jpg', 'jpeg', 'png'], key="profile")
    if profile_photo:
        profile_img = Image.open(profile_photo)
        st.image(profile_img, width=180)
        profile_array = np.array(profile_img)
        if len(profile_array.shape) == 2:
            profile_array = cv2.cvtColor(profile_array, cv2.COLOR_GRAY2RGB)
        elif profile_array.shape[2] == 4:
            profile_array = cv2.cvtColor(profile_array, cv2.COLOR_RGBA2RGB)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖼️ Recent Posts (Optional)</div>', unsafe_allow_html=True)
    post_photos = st.file_uploader("Upload post images for face matching", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if post_photos:
        st.success(f"✅ {len(post_photos)} images uploaded")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Account Metadata</div>', unsafe_allow_html=True)
    followers = st.number_input("👥 Followers", min_value=0, value=1000, step=100)
    following = st.number_input("👤 Following", min_value=0, value=500, step=100)
    posts = st.number_input("📝 Total Posts", min_value=0, value=100, step=50)
    bio = st.text_area("📝 Bio", height=100, placeholder="Enter profile bio...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    analyze_btn = st.button("🔍 ANALYZE PROFILE", use_container_width=True)

# ========== ANALYSIS & RESULTS ==========
if analyze_btn and profile_photo:
    photo_model, face_matcher = load_models()
    
    progress = st.progress(0)
    status = st.empty()
    
    # Module 1
    status.info("📸 Module 1/4: Analyzing photo...")
    profile_resized = cv2.resize(profile_array, (224, 224)) / 255.0
    photo_score = photo_model.predict(profile_resized)
    photo_score = float(photo_score)
    progress.progress(25)
    time.sleep(0.3)
    
    # Module 2
    status.info("👤 Module 2/4: Face matching...")
    face_score = 0.5
    face_msg = "No post images uploaded"
    
    if post_photos:
        post_arrays = []
        for post in post_photos:
            post_img = Image.open(post)
            post_array = np.array(post_img)
            if len(post_array.shape) == 2:
                post_array = cv2.cvtColor(post_array, cv2.COLOR_GRAY2RGB)
            elif len(post_array.shape) == 3 and post_array.shape[2] == 4:
                post_array = cv2.cvtColor(post_array, cv2.COLOR_RGBA2RGB)
            post_arrays.append(post_array)
        face_score, face_msg, _ = face_matcher.match_profile_with_posts(profile_array, post_arrays)
        face_score = float(face_score)
    progress.progress(50)
    time.sleep(0.3)
    
    # ========== MODULE 3: Behavior Pattern Analysis ==========
    status.info("📊 Module 3/4: Analyzing behavior patterns...")
    
    # Calculate ratios
    if following > 0:
        follower_ratio = followers / following
    else:
        follower_ratio = followers
    
    if posts > 0:
        follower_post_ratio = followers / posts
        following_load = following / posts
    else:
        follower_post_ratio = followers
        following_load = following
    
    # Count bio links
    bio_link_count = bio.lower().count('http') + bio.lower().count('www.') + bio.lower().count('.com')
    
    # Suspicious score calculation
    suspicious_score = 0
    suspicious_reasons = []
    
    # 1. Follower/Following ratio
    if follower_ratio > 10:
        suspicious_score += 2
        suspicious_reasons.append(f"Extremely high follower/following ratio ({follower_ratio:.1f}:1)")
    elif follower_ratio > 5:
        suspicious_score += 1.5
        suspicious_reasons.append(f"High follower/following ratio ({follower_ratio:.1f}:1)")
    elif follower_ratio < 0.1:
        suspicious_score += 2
        suspicious_reasons.append(f"Very low follower/following ratio ({follower_ratio:.2f}:1)")
    elif follower_ratio < 0.5:
        suspicious_score += 1
        suspicious_reasons.append(f"Low follower/following ratio ({follower_ratio:.2f}:1)")
    
    # 2. Follower to Post ratio
    if follower_post_ratio > 1000 and posts < 50:
        suspicious_score += 2
        suspicious_reasons.append(f"Suspicious: {int(follower_post_ratio)} followers per post")
    elif follower_post_ratio > 500:
        suspicious_score += 1.5
        suspicious_reasons.append(f"High followers per post ratio")
    elif follower_post_ratio > 200:
        suspicious_score += 0.5
    
    # 3. Following load
    if following_load > 50 and posts < 20:
        suspicious_score += 1.5
        suspicious_reasons.append(f"Following {int(following_load)} accounts per post")
    elif following_load > 20:
        suspicious_score += 0.5
    
    # 4. Bio links
    if bio_link_count > 1:
        suspicious_score += 1.5
        suspicious_reasons.append(f"Multiple external links ({bio_link_count})")
    elif bio_link_count > 0:
        suspicious_score += 0.5
        suspicious_reasons.append(f"Contains external link")
    
    # 5. Account type checks
    if followers < 100 and following > 1000:
        suspicious_score += 1.5
        suspicious_reasons.append("Very low followers, high following")
    
    if posts < 5 and followers > 10000:
        suspicious_score += 2
        suspicious_reasons.append("Very few posts but many followers")
    
    # Convert to ratio_score (0-1 scale)
    ratio_score = min(suspicious_score / 8, 1.0)
    
    # Generate message
    if suspicious_score >= 4:
        growth_msg = "🚨 " + " | ".join(suspicious_reasons[:3])
    elif suspicious_score >= 2:
        growth_msg = "⚠️ " + " | ".join(suspicious_reasons[:3])
    else:
        growth_msg = "✅ Normal account behavior"
    
    ratio_score = float(ratio_score)
    progress.progress(75)
    time.sleep(0.3)
    
    # ========== MODULE 4: Bio Analysis ==========
    status.info("📝 Module 4/4: Analyzing bio content...")
    
    bio_score = 0.0
    bio_warnings = []
    
    suspicious_keywords = [
        'click', 'link', 'free', 'money', 'earn', 'win', 'prize', 'gift', 
        'giveaway', 'offer', 'discount', 'promotion', 'promo', 'deal', 
        'cash', 'pay', 'crypto', 'bitcoin', 'investment', 'profit', 'income',
        'f4f', 'l4l', 's4s', 'follow back', 'follow4follow', 'like4like',
        'dm', 'bio link', 'link in bio', 'swipe up', 'bots', 'auto',
        'buy', 'sell', 'cheap', 'hack', 'password', 'verify', 'urgent'
    ]
    
    if not bio or bio.strip() == "":
        bio_score = 0.6
        bio_warnings.append("❌ Empty bio")
        bio_msg = "⚠️ Empty bio - High risk"
    else:
        bio_lower = bio.lower()
        keyword_matches = []
        
        for word in suspicious_keywords:
            if word in bio_lower:
                bio_score += 0.08
                keyword_matches.append(word)
        
        if bio_link_count > 1:
            bio_score += 0.2
            bio_warnings.append(f"❌ Excessive links ({bio_link_count})")
        elif bio_link_count > 0:
            bio_score += 0.1
            bio_warnings.append(f"⚠️ Contains link")
        
        if bio.isupper() and len(bio) > 10:
            bio_score += 0.15
            bio_warnings.append("⚠️ All caps text")
        
        if len(bio) < 10:
            bio_score += 0.1
            bio_warnings.append("⚠️ Very short bio")
        
        bio_score = min(bio_score, 1.0)
        
        if bio_score < 0.2:
            bio_msg = "✅ Clean bio"
        elif bio_score < 0.4:
            bio_msg = "✅ Bio looks normal"
        elif bio_score < 0.6:
            bio_msg = f"⚠️ {len(keyword_matches)} suspicious keyword(s)"
        else:
            bio_msg = f"❌ Very suspicious bio - {len(keyword_matches)} red flags"
    
    bio_score = float(bio_score)
    progress.progress(100)
    time.sleep(0.3)
    status.empty()
    
    # ========== ENSURE SCORES ARE IN VALID RANGE ==========
    photo_score = max(0.0, min(1.0, photo_score))
    face_score = max(0.0, min(1.0, face_score))
    ratio_score = max(0.0, min(1.0, ratio_score))
    bio_score = max(0.0, min(1.0, bio_score))
    
    # ========== FINAL SCORE CALCULATION ==========
    final_score = (
        weights['photo'] * photo_score +
        weights['face'] * face_score +
        weights['ratio'] * ratio_score +
        weights['bio'] * bio_score
    )
    
    final_score = min(max(final_score, 0), 1)
    final_score = float(final_score)
    
    # ========== IMPERSONATION DETECTION ==========
    impersonation_note = ""
    
    # Case 1: AI/Fake Face (photo_score > 0.6) = High Risk
    if photo_score > 0.6:
        final_score = max(final_score, 0.75)
        impersonation_note = "🚨 AI-generated or fake profile photo detected"
    
    # Case 2: Real face but different person (impersonation)
    elif photo_score < 0.3 and face_score > 0.6:
        final_score = max(final_score, 0.75)
        impersonation_note = "⚠️ Real photo but identity mismatch - Possible impersonation"
    
    # Case 3: Real face with partial mismatch
    elif photo_score < 0.3 and face_score > 0.4:
        final_score = max(final_score, 0.55)
        impersonation_note = "⚠️ Partial identity mismatch - Review recommended"
    
    final_score = min(max(final_score, 0), 1)
    
    # Risk Level
    if final_score <= 0.30:
        risk_emoji = "🟢"
        risk_level = "LOW RISK"
    elif final_score <= 0.60:
        risk_emoji = "🟡"
        risk_level = "MEDIUM RISK"
    else:
        risk_emoji = "🔴"
        risk_level = "HIGH RISK"
    
    # Save history
    st.session_state.analysis_history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'platform': platform,
        'final_score': final_score,
        'risk_level': risk_level
    })
    
    # ========== DISPLAY RESULTS ==========
    st.markdown("---")
    st.markdown("### 📈 Risk Assessment")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Risk Card
    if final_score <= 0.30:
        st.markdown('<div class="risk-low">', unsafe_allow_html=True)
    elif final_score <= 0.60:
        st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-high">', unsafe_allow_html=True)
    
    st.markdown(f'<div class="risk-score">{risk_emoji} {risk_level}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-label">Risk Score: {final_score:.3f} / 1.00</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.progress(final_score)
    
    # Module Scores
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Module Performance")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{photo_score:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📸 Photo</div>', unsafe_allow_html=True)
        if photo_score < 0.3:
            st.markdown('<div class="metric-label" style="color:#28a745;">✅ Real Face</div>', unsafe_allow_html=True)
        elif photo_score < 0.6:
            st.markdown('<div class="metric-label" style="color:#ffc107;">⚠️ Suspicious</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-label" style="color:#dc3545;">❌ Fake/AI Face</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_s2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{face_score:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">👤 Face Match</div>', unsafe_allow_html=True)
        # ========== ANGLE VARIATION DETECTION DISPLAY ==========
        if face_score < 0.28:
            st.markdown('<div class="metric-label" style="color:#28a745;">✅ Same Person</div>', unsafe_allow_html=True)
        elif face_score < 0.48:
            st.markdown('<div class="metric-label" style="color:#17a2b8;">🔄 Same Person (Angle Variation)</div>', unsafe_allow_html=True)
        elif face_score < 0.65:
            st.markdown('<div class="metric-label" style="color:#ffc107;">⚠️ Possible Same Person</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-label" style="color:#dc3545;">❌ Different Person</div>', unsafe_allow_html=True)
        if not post_photos:
            st.markdown('<div class="metric-label" style="color:#666;">(No post images)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_s3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ratio_score:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📊 Behavior</div>', unsafe_allow_html=True)
        if ratio_score < 0.3:
            st.markdown('<div class="metric-label" style="color:#28a745;">✅ Normal</div>', unsafe_allow_html=True)
        elif ratio_score < 0.6:
            st.markdown('<div class="metric-label" style="color:#ffc107;">⚠️ Unusual</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-label" style="color:#dc3545;">❌ Suspicious</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_s4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{bio_score:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📝 Bio</div>', unsafe_allow_html=True)
        if bio_score < 0.3:
            st.markdown('<div class="metric-label" style="color:#28a745;">✅ Clean</div>', unsafe_allow_html=True)
        elif bio_score < 0.6:
            st.markdown('<div class="metric-label" style="color:#ffc107;">⚠️ Suspicious</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-label" style="color:#dc3545;">❌ Spam</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Score Interpretation
    st.markdown("---")
    st.markdown("### 📋 Score Interpretation")
    
    col_int1, col_int2 = st.columns(2)
    
    with col_int1:
        st.markdown(f"**📸 Photo Score: {photo_score:.2f}**")
        if photo_score < 0.3:
            st.markdown("✅ **Real Face** - Profile photo appears genuine")
        elif photo_score < 0.6:
            st.markdown("⚠️ **Suspicious** - Photo may be edited or AI generated")
        else:
            st.markdown("❌ **Fake/AI Face** - High probability of fake profile photo")
        
        st.markdown(f"**👤 Face Match: {face_score:.2f}**")
        if not post_photos:
            st.markdown("ℹ️ **No posts uploaded** - Face matching skipped")
        elif face_score < 0.28:
            st.markdown("✅ **Same Person** - Profile matches post images perfectly")
        elif face_score < 0.48:
            st.markdown("🔄 **Same Person (Angle Variation)** - Different angle/pose detected, but likely same person")
        elif face_score < 0.65:
            st.markdown("⚠️ **Possible Same Person** - Some identity inconsistency")
        else:
            st.markdown("❌ **Different Person** - Identity mismatch detected - possible impersonation")
    
    with col_int2:
        st.markdown(f"**📊 Behavior Score: {ratio_score:.2f}**")
        if ratio_score < 0.3:
            st.markdown("✅ **Normal** - Account behavior appears genuine")
        elif ratio_score < 0.6:
            st.markdown("⚠️ **Unusual** - Some suspicious patterns detected")
        else:
            st.markdown("❌ **Suspicious** - Multiple red flags in behavior")
        if suspicious_reasons:
            st.caption(f"🚩 {suspicious_reasons[0][:60]}...")
        
        st.markdown(f"**📝 Bio Score: {bio_score:.2f}**")
        if bio_score < 0.3:
            st.markdown("✅ **Clean** - Bio looks normal")
        elif bio_score < 0.6:
            st.markdown("⚠️ **Suspicious** - Contains promotional keywords")
        else:
            st.markdown("❌ **Spam** - Bio has spam indicators")
        if bio_warnings:
            st.caption(f"🚩 {bio_warnings[0][:60]}...")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    if impersonation_note:
        st.warning(impersonation_note)
    
    if final_score <= 0.30:
        st.success("✅ **Profile appears genuine.** No action required.")
        st.info("💡 **Tip:** Continue monitoring for any unusual activity.")
    elif final_score <= 0.60:
        st.warning("⚠️ **Moderate risk detected.** Consider verifying the profile manually.")
        if photo_score > 0.5:
            st.write("• 🔍 Profile photo appears suspicious - consider reverse image search")
        if face_score > 0.55 and post_photos:
            st.write("• 👤 Identity mismatch detected between profile and posts")
        if ratio_score > 0.5:
            st.write("• 📊 Suspicious behavior pattern detected")
            for reason in suspicious_reasons[:2]:
                st.write(f"  - {reason}")
        if bio_score > 0.5:
            st.write("• 📝 Bio contains suspicious keywords or links")
            for warning in bio_warnings[:2]:
                st.write(f"  - {warning}")
        st.info("💡 **Action:** Verify identity through direct message or other social platforms.")
    else:
        st.error("🚨 **High risk detected.** This profile shows strong indicators of being fake/impersonation.")
        st.write("**Recommended actions:**")
        st.write("• 📢 Report the profile to the platform immediately")
        st.write("• 🔒 Do not engage or share personal information")
        st.write("• ✅ Verify identity through alternative channels")
        st.write("• 🚫 Block and restrict the profile")
    
    # Download Report
    report_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'platform': platform,
        'final_score': final_score,
        'risk_level': risk_level,
        'photo_score': photo_score,
        'face_score': face_score,
        'ratio_score': ratio_score,
        'bio_score': bio_score,
        'suspicious_reasons': suspicious_reasons,
        'bio_warnings': bio_warnings
    }
    st.download_button(
        label="📥 Download Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

elif analyze_btn and not profile_photo:
    st.error("❌ Please upload a profile photo first!")

# ========== HISTORY ==========
if st.session_state.analysis_history:
    st.markdown("---")
    st.markdown("### 📜 Recent History")
    for h in st.session_state.analysis_history[-5:]:
        st.write(f"🕐 {h['timestamp']} | 🌐 {h['platform']} | Score: {h['final_score']:.2f} | {h['risk_level']}")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p>🚀 Fake Profile Detection System | AI-Powered | Multi-Factor Analysis</p>
</div>
""", unsafe_allow_html=True)