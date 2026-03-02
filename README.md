# ⚡ TransformerGuard AI — Streamlit Deployment Guide

## 🚀 Deploy to Streamlit Cloud (Free, 5 minutes)

### Step 1 — Push to GitHub
```bash
# Create a new repo on github.com, then:
git init
git add app.py requirements.txt
git commit -m "TransformerGuard AI - initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/transformerguard-ai.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **"Deploy"** ✅

Your app will be live at:
`https://YOUR_USERNAME-transformerguard-ai-app-XXXX.streamlit.app`

---

## 📁 File Structure
```
your-repo/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

## 📊 CSV Input Format
The app accepts CSV files with these columns:
- `H2_mean`, `H2_std`, `H2_max`, `H2_min`, `H2_late_slope`, `H2_early_late_ratio`, `H2_variance_growth`, `H2_max_rate`, `H2_cross_time`
- Same pattern for `CO_`, `C2H4_`, `C2H2_`
- `ratio_C2H2_C2H4`, `ratio_H2_CO`, `ratio_CO_C2H4`
- `health_index`

Optional: `Transformer_ID`, `FDD_Label`, `RUL_Label`

## 🔧 To Add Your Trained Model
Replace the `predict_from_features()` function in `app.py` with:
```python
import pickle

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def predict_from_features(df_feat):
    model = load_model()
    preds = model['fdd'].predict(X)
    probas = model['fdd'].predict_proba(X)
    rul_preds = model['rul'].predict(X)
    return preds, probas, rul_preds
```

## 📬 Contact
S. Kiran Krishnan — update contact details in app.py footer section
