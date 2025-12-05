import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi CT Scan Ginjal", page_icon="🩺")

st.title("🩺 Deteksi Kelainan Ginjal (CT Scan)")
st.write("Unggah gambar CT Scan Ginjal untuk mendeteksi: **Normal, Cyst, Tumor, atau Stone**.")

# --- 2. Definisi Label Kelas ---
# PENTING: Pastikan urutan ini sesuai dengan dataset.class_to_idx di notebook Anda.
# Biasanya urutan alfabetis:
CLASS_NAMES = ['Normal', 'Cyst', 'Tumor', 'Stone']

# --- 3. Fungsi Load Model ---
@st.cache_resource # Cache agar model tidak diload ulang setiap kali ada interaksi
def load_model():
    # Load arsitektur ResNet18
    model = models.resnet18(pretrained=False)
    
    # Sesuaikan layer terakhir (Fully Connected) untuk 4 kelas
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Load bobot yang sudah dilatih (pastikan file ada di folder yang sama)
    try:
        model.load_state_dict(torch.load('model_kidney.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("File 'model_kidney.pth' tidak ditemukan. Pastikan sudah diupload.")
        return None
    
    model.eval() # Set ke mode evaluasi
    return model

model = load_model()

# --- 4. Preprocessing Gambar ---
def process_image(image):
    # Transformasi standar untuk ResNet (sesuai training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # Resize ke ukuran input standar ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pastikan gambar RGB (3 channel)
    image = image.convert('RGB')
    
    # Terapkan transformasi dan tambah batch dimension (1, 3, 224, 224)
    img_tensor = preprocess(image).unsqueeze(0)
    return img_tensor

# --- 5. UI Upload & Prediksi ---
uploaded_file = st.file_uploader("Pilih gambar CT Scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if st.button('Analisis Gambar'):
        if model is not None:
            with st.spinner('Sedang menganalisis...'):
                try:
                    # Proses gambar
                    img_tensor = process_image(image)
                    
                    # Prediksi
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probs, 1)
                    
                    # Ambil hasil
                    prediction = CLASS_NAMES[predicted_idx.item()]
                    score = confidence.item() * 100
                    
                    # Tampilkan Hasil
                    st.success(f"Hasil Prediksi: **{prediction}**")
                    st.info(f"Tingkat Keyakinan: {score:.2f}%")
                    
                    # Tampilkan detail probabilitas semua kelas
                    st.write("---")
                    st.write("Detail Probabilitas:")
                    probs_np = probs.numpy()[0]
                    for i, name in enumerate(CLASS_NAMES):
                        st.progress(float(probs_np[i]), text=f"{name}: {probs_np[i]*100:.1f}%")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")