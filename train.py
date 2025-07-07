import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# 1️⃣ 경로 & 파라미터
# -----------------------------

data_dir = '/home/aedu19/바탕화면/output/cifar_dataset'
img_size = (128, 128)
batch_size = 32
epochs = 30

# -----------------------------
# 2️⃣ ImageDataGenerator (증강 강화!)
# -----------------------------

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("클래스 인덱스:", train_gen.class_indices)

# -----------------------------
# 3️⃣ 클래스 가중치 계산
# -----------------------------

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# 4️⃣ CNN 모델 (ConvBlock 개선 + GAP + Dropout ↑)
# -----------------------------

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (5,5), activation='relu'),  # (5,5) 큰 패턴 블록 추가
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),   # Flatten 대신 GAP
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),               # Dropout 조금 더 ↑
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # LR 줄임
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 5️⃣ EarlyStopping
# -----------------------------

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# 6️⃣ 학습
# -----------------------------

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# -----------------------------
# 7️⃣ 결과 시각화
# -----------------------------

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.savefig('result_train_128_finetune.png')

model.save('bird_airplane_people_model_128_finetune.keras')

