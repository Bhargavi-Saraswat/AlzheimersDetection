# src/test_model.py
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths & Config ─────────────────────────────────────────────
MODEL_PATH  = 'saved_model/best_model.h5'
TEST_DIR    = 'data_split/test'
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 16
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# ── Load model ─────────────────────────────────────────────────
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# ── Test data generator ────────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
print("Generator class indices:", test_generator.class_indices)

# ── Predict ────────────────────────────────────────────────────
print("\n🔄 Running predictions...")
test_generator.reset()
preds  = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# ── Classification report ──────────────────────────────────────
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── Confusion matrix ───────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("🧠 Confusion Matrix: Alzheimer's Detection")
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.show()
print("📊 Saved → results/confusion_matrix.png")