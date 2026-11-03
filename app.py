import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import requests
import easyocr
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoModel, AutoProcessor, CLIPModel
import torchvision.transforms as transforms
import torchvision.models as models

st.set_page_config(
    page_title="Dual-Mode Smart Multimodal Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 32px; font-weight: bold; color: #FF9900; text-align: center; }
    .sub-header { font-size: 15px; color: #AAAAAA; text-align: center; margin-bottom: 20px; }
    .price-card { background: linear-gradient(135deg, #11998e, #38ef7d); padding: 20px; border-radius: 12px; text-align: center; color: white; margin-top: 15px; }
    .vision-card { background: linear-gradient(135deg, #8E2DE2, #4A00E0); padding: 20px; border-radius: 12px; text-align: center; color: white; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Dual-Mode E-Commerce Price Intelligence Engine</div>', unsafe_allow_html=True)

# Artifact Loader
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/multimodal_fusion_model.pkl")
    
    # Text BERT
    bert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    bert_model.eval()
    
    # Vision ResNet50
    resnet_weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=resnet_weights)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    # Zero-Shot Vision Classification Engine (CLIP)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # EasyOCR Reader Engine
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    
    return preprocessor, model, bert_tokenizer, bert_model, resnet, clip_model, clip_processor, ocr_reader

preprocessor, model, bert_tokenizer, bert_model, resnet, clip_model, clip_processor, ocr_reader = load_artifacts()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app_mode = st.radio(
    "Choose Intelligence Engine Mode:",
    ["🎛️ Mode 1: Multimodal Fusion (Tabular Specs + NLP Text + Image)", "🖼️ Mode 2: AI OCR + Vision Auto Valuation (Image Only)"],
    horizontal=True
)

st.divider()

# ==============================================================================
# MODE 1: MULTIMODAL FUSION
# ==============================================================================
if "Mode 1" in app_mode:
    st.subheader("🎛️ Mode 1: Complete Feature Fusion Predictor")
    
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("##### 📋 Step 1: Product Specifications")
        category = st.selectbox("Product Category", ["Smartphones", "Audio", "Wearables", "TV", "Accessories"])
        
        if category == "Smartphones":
            brand_options = ["Apple", "Samsung", "OnePlus", "Redmi", "Realme", "Enter Custom Brand..."]
            default_spec = "15 5G (16GB RAM, 512GB Storage)"
            default_mrp = 68999.0
        elif category == "Audio":
            brand_options = ["boAt", "Sony", "JBL", "OnePlus", "Sennheiser", "Enter Custom Brand..."]
            default_spec = "Bullets Z2 Wireless Bluetooth Earbuds"
            default_mrp = 2299.0
        elif category == "Wearables":
            brand_options = ["Noise", "Fire-Boltt", "Apple", "Samsung", "boAt", "Enter Custom Brand..."]
            default_spec = "ColorFit Pulse 2 Max Smartwatch"
            default_mrp = 4999.0
        elif category == "TV":
            brand_options = ["Mi", "Samsung", "LG", "Sony", "OnePlus", "Enter Custom Brand..."]
            default_spec = "5A Series 32 inches HD Ready Smart Android LED TV"
            default_mrp = 14999.0
        else:
            brand_options = ["Logitech", "HP", "Dell", "SanDisk", "TP-Link", "Enter Custom Brand..."]
            default_spec = "Wireless Optical Mouse M235"
            default_mrp = 999.0

        c1, c2 = st.columns(2)
        with c1:
            selected_brand = st.selectbox("Brand Name", brand_options)
            brand = st.text_input("Custom Brand Name", "Nothing") if selected_brand == "Enter Custom Brand..." else selected_brand
        with c2:
            spec_details = st.text_input("Model & Spec Details", default_spec)
        
        full_title = f"{brand} {spec_details}"
        actual_price = st.number_input("Original MRP (₹)", min_value=100.0, value=default_mrp, step=500.0)

    with col2:
        st.markdown("##### 🖼️ Step 2: Visual Spec Input")
        image_source = st.radio("Image Input Mode", ["Upload Local File", "Use Online Image URL"], horizontal=True)
        
        loaded_img = None
        if image_source == "Upload Local File":
            uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"], key="mode1_upload")
            if uploaded_file:
                loaded_img = Image.open(uploaded_file).convert('RGB')
                st.image(loaded_img, caption="Product Preview", width=230)
        else:
            img_url = st.text_input("Product Image URL", "https://m.media-amazon.com/images/I/81Lm92pW3UL._AC_UL320_.jpg", key="mode1_url")
            if img_url:
                try:
                    res = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                    loaded_img = Image.open(BytesIO(res.content)).convert('RGB')
                    st.image(loaded_img, caption="Product Preview", width=230)
                except Exception:
                    st.info("ℹ️ Using ResNet fallback zero vector.")

    if st.button("🚀 Calculate Market Offer Price (Mode 1)", use_container_width=True):
        with st.spinner("Analyzing Tabular + BERT NLP + ResNet Vision Features..."):
            default_rating, default_ratings_count = 4.1, 1174.0
            
            raw_tabular = pd.DataFrame([{
                'name': full_title,
                'custom_sub_category': category,
                'ratings': default_rating,
                'no_of_ratings': default_ratings_count,
                'actual_price': actual_price
            }])
            tab_features = preprocessor.transform(raw_tabular)
            if hasattr(tab_features, "toarray"):
                tab_features = tab_features.toarray()
                
            inputs = bert_tokenizer([full_title], padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = bert_model(**inputs)
                token_embeddings = outputs[0]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                bert_features = (sum_embeddings / sum_mask).numpy()

            if loaded_img is not None:
                try:
                    tensor_img = transform(loaded_img).unsqueeze(0)
                    with torch.no_grad():
                        resnet_features = resnet(tensor_img).squeeze().numpy().reshape(1, -1)
                except Exception:
                    resnet_features = np.zeros((1, 2048))
            else:
                resnet_features = np.zeros((1, 2048))

            fused_features = np.hstack([tab_features, bert_features, resnet_features])
            predicted_price = float(model.predict(fused_features)[0])
            predicted_price = max(0.0, min(predicted_price, actual_price))
            predicted_discount_pct = max(0.0, ((actual_price - predicted_price) / actual_price) * 100)
            savings = actual_price - predicted_price

        st.balloons()
        m1, m2, m3 = st.columns(3)
        m1.metric("Original MRP", f"₹{actual_price:,.2f}")
        m2.metric("Predicted Selling Price", f"₹{predicted_price:,.2f}", delta=f"-{predicted_discount_pct:.1f}%")
        m3.metric("Customer Savings", f"₹{savings:,.2f}")
        
        st.markdown(f"""
            <div class="price-card">
                <h2>🎯 Recommended Selling Price: ₹{predicted_price:,.2f}</h2>
                <p>Engineered via Multimodal Features: Tabular Specs + BERT Text + ResNet Visual Geometry</p>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# MODE 2: ADVANCED OCR + ZERO-SHOT VISION INTELLIGENCE (WITH FINE-GRAINED SERIES RULES)
# ==============================================================================
else:
    st.subheader("🖼️ Mode 2: AI OCR + Vision Automated Product & Price Estimator")
    st.markdown("Upload any product image! EasyOCR reads text details and CLIP Vision identifies product visual geometry.")

    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        uploaded_vision_file = st.file_uploader("Upload Any Product Image", type=["jpg", "png", "jpeg"], key="mode2_upload")
        loaded_vision_img = None
        if uploaded_vision_file:
            loaded_vision_img = Image.open(uploaded_vision_file).convert('RGB')
            st.image(loaded_vision_img, caption="Uploaded Product Image", width=280)

    with v_col2:
        if loaded_vision_img is not None:
            if st.button("🤖 Run OCR & Vision Auto-Valuation", use_container_width=True):
                with st.spinner("Extracting OCR Text & Vision Feature Vectors..."):
                    
                    # 1. EasyOCR Text Extraction
                    img_np = np.array(loaded_vision_img)
                    ocr_results = ocr_reader.readtext(img_np, detail=0)
                    extracted_text = " ".join(ocr_results).strip()
                    
                    # 2. CLIP Zero-Shot Classification
                    candidate_labels = [
                        "Apple AirPods wireless earbuds", 
                        "Bluetooth in-ear headphones",
                        "Apple iPhone smartphone", 
                        "Samsung Galaxy flagship smartphone", 
                        "Smartwatch fitness tracker", 
                        "Smart LED Television",
                        "Computer Accessories Mouse Keyboard"
                    ]
                    
                    inputs_clip = clip_processor(text=candidate_labels, images=loaded_vision_img, return_tensors="pt", padding=True)
                    outputs_clip = clip_model(**inputs_clip)
                    probs = outputs_clip.logits_per_image.softmax(dim=1)
                    best_label_idx = torch.argmax(probs).item()
                    vision_detected_label = candidate_labels[best_label_idx]

                    if extracted_text and len(extracted_text) > 3:
                        title_text = f"{extracted_text} {vision_detected_label}"
                        detection_source = "EasyOCR Text Engine + CLIP Vision"
                    else:
                        title_text = vision_detected_label
                        detection_source = "CLIP Vision Neural Classifier"

                    t_lower = title_text.lower()

                    # 3. Smart Fine-Grained Series Mapping (Budget vs Ultra-Flagship)
                    if any(k in t_lower for k in ['airpods', 'buds', 'headphone', 'earphone', 'bluetooth']):
                        dummy_cat = "Audio"
                        dummy_mrp = 19999.0 if "airpods" in t_lower else 2499.0
                    
                    elif any(k in t_lower for k in ['iphone', 'galaxy', 'phone', 'smartphone', 'samsung', 'redmi', 'realme']):
                        dummy_cat = "Smartphones"
                        
                        # Ultra-Flagship Series (₹1 Lakh+)
                        if any(k in t_lower for k in ['ultra', 'pro max', 's23 ultra', 's24 ultra', 'fold', 'flip', 'iphone 14 pro', 'iphone 15 pro', 'iphone 16 pro']):
                            dummy_mrp = 139999.0
                        # Mid-Range Series (₹25k - ₹40k)
                        elif any(k in t_lower for k in ['s23', 's24', 's22', 'oneplus 11', 'oneplus 12', 'iphone 13', 'iphone 14', 'iphone 15']):
                            dummy_mrp = 64999.0
                        # Budget Series (₹10k - ₹15k) - e.g. M07, M12, A-series, Redmi
                        elif any(k in t_lower for k in ['m07', 'm12', 'm14', 'm34', 'a04', 'a14', 'redmi', 'realme', 'c55']):
                            dummy_mrp = 13999.0
                        else:
                            dummy_mrp = 24999.0

                    elif any(k in t_lower for k in ['watch', 'smartwatch']):
                        dummy_cat = "Wearables"
                        dummy_mrp = 4999.0
                    elif any(k in t_lower for k in ['tv', 'television', 'led']):
                        dummy_cat = "TV"
                        dummy_mrp = 24999.0
                    else:
                        dummy_cat = "Accessories"
                        dummy_mrp = 1499.0

                    # 4. Multimodal Inference
                    tensor_img = transform(loaded_vision_img).unsqueeze(0)
                    with torch.no_grad():
                        resnet_features = resnet(tensor_img).squeeze().numpy().reshape(1, -1)

                    raw_tabular = pd.DataFrame([{
                        'name': title_text,
                        'custom_sub_category': dummy_cat,
                        'ratings': 4.3,
                        'no_of_ratings': 2500.0,
                        'actual_price': dummy_mrp
                    }])
                    tab_features = preprocessor.transform(raw_tabular)
                    if hasattr(tab_features, "toarray"):
                        tab_features = tab_features.toarray()

                    inputs_bert = bert_tokenizer([title_text], padding=True, truncation=True, max_length=128, return_tensors="pt")
                    with torch.no_grad():
                        outputs_bert = bert_model(**inputs_bert)
                        token_embeddings = outputs_bert[0]
                        attention_mask = inputs_bert['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                        bert_features = (sum_embeddings / sum_mask).numpy()

                    fused_features = np.hstack([tab_features, bert_features, resnet_features])
                    base_predicted = float(model.predict(fused_features)[0])
                    
                    lower_bound = max(500.0, base_predicted * 0.88)
                    upper_bound = base_predicted * 1.12

                st.success(f"🎯 **Detected Intelligence Source:** `{detection_source}`")
                st.markdown("### 📊 AI System Extraction Summary")
                if extracted_text:
                    st.write(f"• **OCR Extracted Text from Image:** `{extracted_text}`")
                st.write(f"• **Vision Transformer Classifier Label:** `{vision_detected_label}`")
                st.write(f"• **Inferred Category & Series Tier:** `{dummy_cat}` (Base MRP: ₹{dummy_mrp:,.2f})")
                
                st.markdown(f"""
                    <div class="vision-card">
                        <h3>🏷️ Estimated Market Price Range</h3>
                        <h2>₹{lower_bound:,.2f} — ₹{upper_bound:,.2f}</h2>
                        <p>Recommended Optimal Selling Price: <b>₹{base_predicted:,.2f}</b></p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload an image on the left. AI Vision Model & OCR Engine will automatically read text, classify the product, and predict its market price!")