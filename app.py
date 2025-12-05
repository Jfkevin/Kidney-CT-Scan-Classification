import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Deteksi Kelainan Ginjal CT-Scan", page_icon="🩺")

# --- 1. Definisi Kelas ---
CLASS_NAMES = ['Normal', 'Cyst', 'Tumor', 'Stone']

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    try:
        state_dict = torch.load('model_kidney.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# --- 3. UI & Prediksi ---
st.title("🩺 Deteksi Kelainan Ginjal Menggunakan Citra CT-Scan")
st.subheader("**Menggunakan Arsitektur CNN ResNet-18**")
st.write("Upload gambar CT Scan untuk mendeteksi: **Normal, Cyst, Tumor, atau Stone**.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar Upload', width=300)
    
    if st.button('Analisis'):
        if model:
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)
            
            # Prediksi
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, 1)
                
            st.success(f"Hasil: **{CLASS_NAMES[idx.item()]}**")
            st.info(f"Keyakinan: {conf.item()*100:.2f}%")