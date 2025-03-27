import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Ayarlar
IMG_SIZE = (224, 224)
CATEGORIES = ['askılı', 'reglan', 'straplez']  # Kategori sırası eğitirkenki sıra olmalı

# Modeli yükle
@st.cache_resource
def load_model_once():
    return load_model("kiyafet_model.h5")

model = load_model_once()

# Başlık
st.title("👕 Kıyafet Tanıma Uygulaması")
st.write("Bir ürün fotoğrafı yükleyin, hangi kategoriye ait olduğunu tahmin edelim.")

# Görsel yükleme alanı
uploaded_file = st.file_uploader("Bir resim yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli yeniden boyutlandır
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    prediction = model.predict(img_array)
    predicted_label = CATEGORIES[np.argmax(prediction)]

    st.success(f"📌 Tahmin Edilen Kategori: **{predicted_label}**")
