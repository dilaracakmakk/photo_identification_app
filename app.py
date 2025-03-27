import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import requests

CATEGORIES = ['askılı', 'reglan', 'straplez']
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_once():
    model_path = "kiyafet_model.h5"
    if not os.path.exists(model_path):
        # Dosyayı ilk çalıştırmada indir
        url = "https://we.tl/t-9xJXTGOezX"  # WeTransfer linkin burada
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

model = load_model_once()

st.title("👕 Kıyafet Tanıma Uygulaması")
st.write("Bir ürün fotoğrafı yükleyin, hangi kategoriye ait olduğunu tahmin edelim.")

uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)

    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_label = CATEGORIES[np.argmax(prediction)]

    st.success(f"📌 Tahmin Edilen Kategori: **{predicted_label}**")
