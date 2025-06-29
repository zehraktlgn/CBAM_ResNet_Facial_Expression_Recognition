import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# 1. Modeli yükle
model = load_model("raf_cbam_model.h5")

# 2. Test verisini hazırla
test_dir = r"C:\Users\zehra\OneDrive\Masaüstü\PYTHON projeleri\RAF_DuyguAnalizi\DATASET\test"

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Önemli: Karışık olmasın, etiket sırası bozulmaz
)

# ✅ 3. Modeli değerlendir
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🎯 TEST DOĞRULUĞU: {test_accuracy * 100:.2f}%")
print(f"❌ TEST KAYBI: {test_loss:.4f}")

# 🔍 4. Tahminleri al
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# 🎯 5. Skorları hesapla
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"\n📊 F1 Skoru (Weighted): {f1:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔁 Recall: {recall:.4f}")

# 🔍 6. İsteğe bağlı: Sınıf bazlı detaylı rapor
print("\n📝 Sınıf Bazlı Rapor:\n")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
