# src/train.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data
from model import build_cnn

os.makedirs('saved_model', exist_ok=True)
os.makedirs('results',     exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────
train_gen, val_gen, test_gen = load_data()
print("Class indices :", train_gen.class_indices)
print("Train samples :", train_gen.samples)
print("Val   samples :", val_gen.samples)
print("Test  samples :", test_gen.samples)

# ── 2. Class weights ───────────────────────────────────────────
labels  = train_gen.classes
weights = compute_class_weight('balanced',
                               classes=np.unique(labels),
                               y=labels)
class_weight_dict = dict(enumerate(weights))
print("\nClass weights (higher = more attention to rare class):")
class_names = list(train_gen.class_indices.keys())
for i, w in class_weight_dict.items():
    print(f"  {class_names[i]}: {w:.4f}")

# ── 3. Build model ─────────────────────────────────────────────
model, base_model = build_cnn(input_shape=(128, 128, 3), num_classes=4)  # ← changed 224 to 128
model.summary()

# ── 4. Callbacks ───────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    'saved_model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ── 5. Phase 1: Top layers only (base frozen) ─────────────────
print("\n====== PHASE 1: Training top layers — 15 epochs ======")
history1 = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ── 6. Phase 2: Unfreeze last 30 layers and fine-tune ─────────
print("\n====== PHASE 2: Fine-tuning — 20 epochs ======")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a much lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ── 7. Evaluate on test set ────────────────────────────────────
print("\n====== FINAL TEST EVALUATION ======")
test_gen.reset()
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# ── 8. Training history plot ───────────────────────────────────
a  = history1.history['accuracy']     + history2.history['accuracy']
va = history1.history['val_accuracy'] + history2.history['val_accuracy']
l  = history1.history['loss']         + history2.history['loss']
vl = history1.history['val_loss']     + history2.history['val_loss']
pb = len(history1.history['accuracy'])  # phase boundary

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(a,  label='Train Accuracy')
plt.plot(va, label='Val Accuracy')
plt.axvline(pb, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Model Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(l,  label='Train Loss')
plt.plot(vl, label='Val Loss')
plt.axvline(pb, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Model Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.savefig('results/training_history.png')
plt.show()
print("📊 Saved → results/training_history.png")