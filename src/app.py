import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from model import SiameseNet
import torch.nn.functional as F
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Signature Verification", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.69   # Final tuned threshold

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = SiameseNet().to(device)

    # Model is inside src/models/
    base_dir = os.path.dirname(os.path.abspath(__file__))   # src folder
    model_path = os.path.join(base_dir, "models", "siamese_vit_cnn.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# IMPORTANT: Load model here (GLOBAL)
model = load_model()

# ---------------- TRANSFORM ----------------
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ---------------- UI ----------------
st.title("✍️ Signature Verification System")
st.markdown("Upload two signature images and verify whether they belong to the same person.")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Signature A", type=["png", "jpg", "jpeg"])

with col2:
    img2_file = st.file_uploader("Upload Signature B", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")

    st.image([img1, img2], caption=["Signature A", "Signature B"], width=200)

    if st.button("🔍 Verify Signatures"):

        x1 = transform(img1).unsqueeze(0).to(device)
        x2 = transform(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            e1, e2 = model(x1, x2)
            dist = F.pairwise_distance(e1, e2).item()

        st.markdown("---")
        st.subheader("🔎 Verification Result")

        st.write(f"**Distance:** `{dist:.4f}`")
        st.write(f"**Threshold:** `{THRESHOLD}`")

        if dist < THRESHOLD:
            st.success("✅ SAME PERSON (GENUINE SIGNATURE)")
        else:
            st.error("❌ FORGERY / DIFFERENT PERSON")

        st.info("Lower distance = more similar signatures. Decision is based on threshold optimized from ROC/EER analysis.")
