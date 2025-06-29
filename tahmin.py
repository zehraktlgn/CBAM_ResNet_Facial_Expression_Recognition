import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# ✅ Modeli yükle (.h5 dosyan bu klasörde olmalı)
model = load_model("raf_cbam_model.h5")

# ✅ Sınıf etiketleri (klasör sırasına göre güncellemek istersen bana yaz)
class_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# ✅ Görsellerin bulunduğu klasör
folder_path = "fotolar"

# ✅ Her görsel için tahmin yap ve sonucu göster
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, file_name)

        # Görseli yükle ve modele uygun hale getir
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin yap
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]

        # Sonucu görselleştir
        plt.imshow(img)
        plt.title(f"Tahmin: {predicted_label}")
        plt.axis("off")
        plt.show()
