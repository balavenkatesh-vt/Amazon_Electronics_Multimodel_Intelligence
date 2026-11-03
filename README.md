# ⚡ Multimodal E-Commerce Product Price Intelligence Engine

An end-to-end Multimodal Artificial Intelligence system designed to predict dynamic product selling prices for e-commerce listings (Amazon Electronics Dataset). The system integrates **Tabular Feature Engineering**, **Contextual NLP Embeddings (Sentence-BERT)**, **Visual Neural Features (ResNet50 CNN)**, and an automated **OCR + Zero-Shot Valuation Engine (EasyOCR & CLIP)** through Late Fusion XGBoost Regression.

---

## 🌟 Key Features & Dual-Mode System

### 🎛️ Mode 1: Complete Multimodal Feature Fusion Predictor
* **Tabular Metadata Processing:** Custom One-Hot Encoding and Robust Scaling for numerical specs, brand categories, ratings, and original MRP.
* **Contextual NLP Embeddings:** Dense 384-dimensional contextual text embeddings extracted using `sentence-transformers/all-MiniLM-L6-v2` from product titles.
* **Deep Visual Embeddings:** 2048-dimensional feature representations extracted via pretrained **ResNet50 CNN** architecture from product visual listings.
* **Late Fusion Regressor:** XGBoost Regressor trained on fused feature matrices achieving **96.06% $R^2$ Score** with complete data-leakage prevention.

### 🖼️ Mode 2: AI Automated OCR + Vision Valuation Engine
* **Optical Character Recognition (EasyOCR):** Automatically extracts printed specs, brand names, and model text from uploaded raw product images.
* **Zero-Shot Visual Classification (OpenAI CLIP):** Classifies unknown product geometry (*Smartphones, AirPods, Smartwatches, TVs*) when visual text is sparse.
* **Fine-Grained Series Heuristics:** Automatically distinguishes budget series (e.g., *Samsung M-series*) from ultra-flagships (*Ultra / Pro Max variants*) to establish dynamic floor price boundaries.

---

## 📊 Model Performance & Metrics

| Metric | Score / Value |
| :--- | :--- |
| **$R^2$ Score (Accuracy)** | **96.06%** |
| **Model Architecture** | Late Fusion XGBoost Regressor |
| **Text Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) |
| **Vision Extractor** | ResNet50 PyTorch CNN |
| **Zero-Shot Classifier** | OpenAI CLIP (`clip-vit-base-patch32`) |
| **OCR Engine** | EasyOCR |

---

## 🛠️ Technology Stack

* **Programming & Environment:** Python 3.10+, PyTorch, Torchvision
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib, Pandas, NumPy
* **Deep Learning & Transformers:** Hugging Face Transformers, Sentence-BERT, CLIP
* **Computer Vision & OCR:** ResNet50, EasyOCR, Pillow (PIL), OpenCV
* **Web Dashboard & Deployment:** Streamlit Framework

---

## 📁 Repository Structure

```text
.
├── app.py                           # Dual-Mode Production Streamlit Dashboard
├── data/
│   ├── splits/                      # Train-Test Data Splits (X_train, X_test)
│   └── embeddings/                  # Saved BERT & ResNet Dense Vectors (.npy)
├── models/
│   ├── preprocessor.pkl             # Fitted ColumnTransformer Artifact
│   └── multimodal_fusion_model.pkl  # Trained XGBoost Late Fusion Model
├── .gitignore                       # Git exclusion rules
├── requirements.txt                 # Project Dependencies
└── README.md                        # Project Documentation

🚀 Quickstart & Local Setup
1. Clone the Repository
Bash
git clone [https://github.com/balavenkatesh-vt/Amazon_Electronics_Multimodel_Intelligence.git](https://github.com/balavenkatesh-vt/Amazon_Electronics_Multimodel_Intelligence.git)
cd Amazon_Electronics_Multimodel_Intelligence
2. Set Up Virtual Environment & Dependencies
Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
3. Launch the Streamlit Intelligence Engine
Bash
python -m streamlit run app.py