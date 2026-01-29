# ✍️ Offline Signature Verification using Hybrid CNN–ViT Siamese Network

This project implements an **offline biometric signature verification system** that determines whether two handwritten signatures belong to the same person or if one is a forgery.

The system uses a **Siamese neural network** with a hybrid **CNN + Vision Transformer (ViT)** encoder and performs verification using **distance-based metric learning**.

---

## 🔍 Problem Statement

Given two signature images:
- ✅ Verify if they belong to the **same individual**
- ❌ Detect **forged or mismatched signatures**

This is a classic **offline signature verification** problem widely used in **KYC, document verification, and identity authentication systems**.

---

## 🧠 Model Architecture

- **CNN (ResNet18)** → captures local stroke-level features  
- **Vision Transformer (ViT-Tiny)** → captures global structural patterns  
- **Siamese Network** → learns embedding similarity  
- **Contrastive Loss** → minimizes distance for genuine pairs and maximizes for forged pairs  

(Signature A & Signature B) ── CNN + ViT ──> Embedding

Distance between embeddings → SAME / FORGERY


---

## 📊 Dataset

- CEDAR-style offline signature dataset  
- Genuine and forged signatures per user  
- Dataset not included in repo due to size/license  

---

## 📈 Evaluation Metrics

The system was evaluated using biometric verification metrics:

| Metric | Value |
|------|------|
| Accuracy | **94.51%** |
| FAR (False Accept Rate) | **5.8%** |
| FRR (False Reject Rate) | **5.19%** |
| EER (Equal Error Rate) | **5.5%** |

- Threshold optimized using **ROC & EER analysis**
- Final decision threshold: **0.69**

---

## 🌐 Web Application (Streamlit)

A production-style **Streamlit web app** is included:

Features:
- Upload two signature images
- Real-time verification
- Distance score + decision
- Tuned threshold display

Run the app:

```bash
streamlit run src/app.py
