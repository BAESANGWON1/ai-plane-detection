import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

# -----------------------------
# ✅ 1️⃣ 경로 & 파라미터
# -----------------------------

base_dir = '/home/aedu19/바탕화면/output/cifar_datatest'
categories = ['airplane', 'bird', 'people']
img_size = (128, 128)

# -----------------------------
# ✅ 2️⃣ 개선된 모델 로드
# -----------------------------

model = tf.keras.models.load_model('bird_airplane_people_model_128_improved.keras')

# -----------------------------
# ✅ 3️⃣ 랜덤 샘플 추출 (테스트셋)
# -----------------------------

num_samples = 16

sample_images = []
true_classes = []

for _ in range(num_samples):
    cls = random.choice(categories)
    cls_path = os.path.join(base_dir, cls)
    img_file = random.choice(os.listdir(cls_path))
    img_path = os.path.join(cls_path, img_file)

    img = Image.open(img_path).resize(img_size)
    sample_images.append(np.array(img))
    true_classes.append(cls)

# -----------------------------
# ✅ 4️⃣ 예측
# -----------------------------

sample_images_np = np.array(sample_images) / 255.0

predictions = model.predict(sample_images_np)

pred_labels = []
for p in predictions:
    class_idx = np.argmax(p)
    pred_labels.append(categories[class_idx])

# -----------------------------
# ✅ 5️⃣ 시각화
# -----------------------------

plt.figure(figsize=(12, 12))

for i in range(num_samples):
    plt.subplot(4, 4, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {true_classes[i]}\nPred: {pred_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('result_test_128_improved.png')
plt.show()

