import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Dropout, Multiply, Add, Reshape, Activation, MaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 📌 CBAM Blok
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape

def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    # 🔵 Channel Attention

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    channel_refined = Multiply()([input_feature, cbam_feature])

    return channel_refined


# 🔧 Model Kurulumu
def build_cbam_resnet(input_shape=(224, 224, 3), num_classes=7):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# 📁 Klasörleri Ayarla
train_dir = r"C:\Users\zehra\OneDrive\Masaüstü\PYTHON projeleri\RAF_DuyguAnalizi\DATASET\train"
val_dir = r"C:\Users\zehra\OneDrive\Masaüstü\PYTHON projeleri\RAF_DuyguAnalizi\DATASET\test"


# 🔄 Veri Artırma
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 🔄 Verileri Yükle
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 🧠 Modeli Derle
model = build_cbam_resnet(input_shape=(224, 224, 3), num_classes=train_generator.num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 📌 Callback’ler
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
    ModelCheckpoint("raf_cbam_model.h5", save_best_only=True)
]

# 🚀 Eğitimi Başlat
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)

# 🎯 Eğitim Sonrası
print("\n✅ Model başarıyla eğitildi ve raf_cbam_model.h5 olarak kaydedildi.")

# 🔍 Eğitim Sonuçlarını Görselleştir
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy per Epoch")
plt.show()
